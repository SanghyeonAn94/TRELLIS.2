"""Local regen + stitch (head + chest) — for GPU 7"""
import os, sys, time
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, '/workspace')

import cv2, imageio, trimesh
import numpy as np, torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

pipeline = Trellis2ImageTo3DPipeline.from_pretrained('/workspace/checkpoints/TRELLIS.2-4B')
pipeline.cuda()

envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('/workspace/assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

image = Image.open('/workspace/assets/inhouse/man.png')
preprocessed = pipeline.preprocess_image(image)

SEED, REGEN_SEED, DS, CFG, PADDING, STEPS = 42, 456, 0.8, 1.0, 3, 20
FEATHER = 2  # blending zone width in voxels
REGEN_TEXTURE = False  # True=shape+texture, False=shape only (keep original texture)
ORIG_QUALITY = dict(
    sparse_structure_sampler_params={'steps': STEPS, 'guidance_strength': 3.0, 'guidance_rescale': 0.7, 'guidance_interval': [0.6, 1.0], 'rescale_t': 5.0},
    shape_slat_sampler_params={'steps': STEPS, 'guidance_strength': 7.5, 'guidance_rescale': 0.5, 'guidance_interval': [0.6, 1.0], 'rescale_t': 3.0},
    tex_slat_sampler_params={'steps': STEPS, 'guidance_strength': 1.0, 'guidance_rescale': 0.0, 'guidance_interval': [0.6, 0.9], 'rescale_t': 3.0},
)

out = '/workspace/outputs/local_regen_stitch'
os.makedirs(out, exist_ok=True)

def export_mesh(mesh, path_prefix):
    mesh.simplify(16777216)
    vid = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
    imageio.mimsave(f'{path_prefix}.mp4', vid, fps=15)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], decimation_target=1000000,
        texture_size=4096, remesh=True, remesh_band=1, remesh_project=0, verbose=True)
    glb.export(f'{path_prefix}.glb')
    print(f'  Exported: {path_prefix}.mp4 + .glb')

# Generate original once
print("Generating original...")
_, (shape_slat, tex_slat, res) = pipeline.run(
    preprocessed, seed=SEED, return_latent=True, preprocess_image=False,
    pipeline_type='1024_1536', denoise_strength=1.0, max_num_tokens=131072, **ORIG_QUALITY)
coarse_grid = res // 16

export_mesh(pipeline.decode_latent(shape_slat, tex_slat, res)[0], f'{out}/original')

shape_std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].cuda()
shape_mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].cuda()
tex_std = torch.tensor(pipeline.tex_slat_normalization['std'])[None].cuda()
tex_mean = torch.tensor(pipeline.tex_slat_normalization['mean'])[None].cuda()
cond = pipeline.get_cond([preprocessed], 1024)

for name, path in [('head', '/workspace/assets/masks/man_mask_head.glb'),
                    ('chest', '/workspace/assets/masks/man_mask_chest.glb')]:
    print(f'\n{"="*60}\nStitch: {name}\n{"="*60}')
    t1 = time.time()

    mask = pipeline.create_voxel_mask_from_mesh(
        np.array(trimesh.load(path, force='mesh').vertices, dtype=np.float32),
        shape_slat.coords, res, radius=1)
    masked_xyz = shape_slat.coords[mask][:, 1:]
    aabb_min = (masked_xyz.min(0).values - PADDING).clamp(min=0)
    aabb_max = (masked_xyz.max(0).values + PADDING).clamp(max=coarse_grid-1)
    xyz = shape_slat.coords[:, 1:]
    in_aabb = ((xyz[:,0]>=aabb_min[0])&(xyz[:,0]<=aabb_max[0])&
               (xyz[:,1]>=aabb_min[1])&(xyz[:,1]<=aabb_max[1])&
               (xyz[:,2]>=aabb_min[2])&(xyz[:,2]<=aabb_max[2]))
    print(f'  AABB voxels: {in_aabb.sum().item()}')

    # Shape regen
    lc = torch.cat([torch.zeros(in_aabb.sum(),1,dtype=torch.int32,device='cuda'), xyz[in_aabb]], 1)
    x0 = (shape_slat.feats[in_aabb] - shape_mean) / shape_std
    fm = pipeline.models['shape_slat_flow_model_1024']
    C = fm.in_channels
    if x0.shape[1]<C: x0 = torch.cat([x0, torch.zeros(x0.shape[0],C-x0.shape[1],device='cuda')],1)
    torch.manual_seed(REGEN_SEED)
    eps = torch.randn_like(x0)
    ni = (1-DS)*x0 + DS*eps
    sv=None
    if coarse_grid>64: sv=pipeline._apply_ntk_rope_scaling(fm, coarse_grid/64)
    try:
        fm.to('cuda')
        ls = pipeline.shape_slat_sampler.sample(fm, SparseTensor(feats=ni,coords=lc), **cond,
            steps=STEPS, guidance_strength=CFG, guidance_rescale=0.5,
            guidance_interval=[0.6,1.0], rescale_t=3.0, start_t=DS, verbose=True,
            tqdm_desc=f"Shape ({name})").samples
        fm.cpu(); ls = ls*shape_std+shape_mean
    finally:
        if sv: pipeline._restore_rope_freqs(fm, sv)

    # Texture regen (optional)
    if REGEN_TEXTURE:
        sn = ls.replace(feats=(ls.feats-shape_mean)/shape_std)
        tfm = pipeline.models['tex_slat_flow_model_1024']
        tsv=None
        if coarse_grid>64: tsv=pipeline._apply_ntk_rope_scaling(tfm, coarse_grid/64)
        try:
            torch.manual_seed(REGEN_SEED+1)
            tnc = tfm.in_channels - sn.feats.shape[1]
            tn = sn.replace(feats=torch.randn(sn.coords.shape[0],tnc).cuda())
            tfm.to('cuda')
            lt = pipeline.tex_slat_sampler.sample(tfm, tn, concat_cond=sn, **cond,
                steps=STEPS, guidance_strength=1.0, guidance_rescale=0.0,
                guidance_interval=[0.6,0.9], rescale_t=3.0, verbose=True,
                tqdm_desc=f"Texture ({name})").samples
            tfm.cpu(); lt = lt*tex_std+tex_mean
        finally:
            if tsv: pipeline._restore_rope_freqs(tfm, tsv)
    else:
        lt = None
        print(f'  Texture: keeping original (REGEN_TEXTURE=False)')

    # Stitch with feathered blending
    # Compute per-voxel distance to AABB boundary (0=boundary, higher=deeper inside)
    xyz_f = xyz.float()
    dist_to_min = xyz_f - aabb_min.unsqueeze(0).float()  # [N, 3]
    dist_to_max = aabb_max.unsqueeze(0).float() - xyz_f   # [N, 3]
    # Min distance to any AABB face (negative = outside)
    min_dist = torch.min(dist_to_min, dist_to_max).min(dim=1).values  # [N]
    # Blend weight: 0 at boundary, 1 at FEATHER voxels inside
    blend_w = (min_dist.clamp(0, FEATHER) / FEATHER).unsqueeze(1)  # [N, 1]
    # Only apply to voxels inside AABB
    blend_w[~in_aabb] = 0.0

    sf = shape_slat.feats.clone()
    sf[in_aabb] = (1 - blend_w[in_aabb]) * sf[in_aabb] + blend_w[in_aabb] * ls.feats
    tf = tex_slat.feats.clone()
    if lt is not None:
        tf[in_aabb] = (1 - blend_w[in_aabb]) * tf[in_aabb] + blend_w[in_aabb] * lt.feats

    n_full = (blend_w[in_aabb] >= 1.0).all(dim=1).sum().item()
    n_blend = in_aabb.sum().item() - n_full
    print(f'  Blend: {n_full} full + {n_blend} feathered voxels (feather={FEATHER})')

    torch.cuda.empty_cache()
    mesh = pipeline.decode_latent(shape_slat.replace(feats=sf), tex_slat.replace(feats=tf), res)[0]
    print(f'  Stitched: {mesh.vertices.shape[0]:,} verts ({time.time()-t1:.1f}s)')
    tag = f'ds{DS}' + ('_shapeonly' if not REGEN_TEXTURE else '')
    export_mesh(mesh, f'{out}/stitched_{name}_{tag}')

print('\nAll Stitch done!')
