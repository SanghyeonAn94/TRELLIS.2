"""
Regen masked region with NEW image conditioning + feather stitch.
Loads pre-saved latents (no re-generation needed).

Usage:
  python scripts/run_regen_with_image.py \
    --latents outputs/promote/latents.pt \
    --mask assets/masks/man_mask_head.glb \
    --new-image assets/promote/inpaint_images/head/head-1.png \
    --output outputs/promote/regen_head1 \
    --ds 0.8 --shape-cfg 3.0 --tex-cfg 3.0 --feather 2
"""
import os, sys, argparse, time
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

parser = argparse.ArgumentParser()
parser.add_argument('--latents', required=True, help='Path to latents.pt from original generation')
parser.add_argument('--mask', required=True, help='GLB mask mesh path')
parser.add_argument('--new-image', required=True, help='New conditioning image for regen')
parser.add_argument('--output', required=True, help='Output prefix (without extension)')
parser.add_argument('--ds', type=float, default=0.8, help='Denoise strength (0=keep original, 1=full regen)')
parser.add_argument('--shape-cfg', type=float, default=3.0, help='Shape guidance strength')
parser.add_argument('--tex-cfg', type=float, default=3.0, help='Texture guidance strength')
parser.add_argument('--feather', type=int, default=2, help='Feather blending width in voxels')
parser.add_argument('--padding', type=int, default=3, help='AABB padding in voxels')
parser.add_argument('--mask-radius', type=int, default=1, help='Mask dilation radius')
parser.add_argument('--seed', type=int, default=456, help='Regen seed')
parser.add_argument('--steps', type=int, default=20, help='Sampling steps')
parser.add_argument('--shape-only', action='store_true', help='Regen shape only, keep original texture')
args = parser.parse_args()

CHECKPOINT = '/workspace/checkpoints/TRELLIS.2-4B'

pipeline = Trellis2ImageTo3DPipeline.from_pretrained(CHECKPOINT)
pipeline.cuda()
print('Pipeline loaded')

envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('/workspace/assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# ── Load pre-saved latents ────────────────────────────────────────
print(f'Loading latents: {args.latents}')
data = torch.load(args.latents, map_location='cuda', weights_only=True)
shape_slat = SparseTensor(feats=data['shape_slat_feats'].cuda(), coords=data['shape_slat_coords'].cuda())
tex_slat = SparseTensor(feats=data['tex_slat_feats'].cuda(), coords=data['tex_slat_coords'].cuda())
res = data['res']
coarse_grid = res // 16
print(f'  res={res}, grid={coarse_grid}³, voxels={shape_slat.coords.shape[0]}')

# ── New image conditioning ────────────────────────────────────────
print(f'New image: {args.new_image}')
new_image = Image.open(args.new_image)
new_preprocessed = pipeline.preprocess_image(new_image)
cond = pipeline.get_cond([new_preprocessed], 1024)

# ── Extract AABB ──────────────────────────────────────────────────
mask_mesh = trimesh.load(args.mask, force='mesh')
mask_verts = np.array(mask_mesh.vertices, dtype=np.float32)
mask = pipeline.create_voxel_mask_from_mesh(mask_verts, shape_slat.coords, res, radius=args.mask_radius)

masked_xyz = shape_slat.coords[mask][:, 1:]
aabb_min = (masked_xyz.min(0).values - args.padding).clamp(min=0)
aabb_max = (masked_xyz.max(0).values + args.padding).clamp(max=coarse_grid - 1)

xyz = shape_slat.coords[:, 1:]
in_aabb = ((xyz[:, 0] >= aabb_min[0]) & (xyz[:, 0] <= aabb_max[0]) &
           (xyz[:, 1] >= aabb_min[1]) & (xyz[:, 1] <= aabb_max[1]) &
           (xyz[:, 2] >= aabb_min[2]) & (xyz[:, 2] <= aabb_max[2]))
print(f'  AABB voxels: {in_aabb.sum().item()}, mask: {mask.sum().item()}')

# ── Normalization constants ───────────────────────────────────────
shape_std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].cuda()
shape_mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].cuda()
tex_std = torch.tensor(pipeline.tex_slat_normalization['std'])[None].cuda()
tex_mean = torch.tensor(pipeline.tex_slat_normalization['mean'])[None].cuda()

# ── Shape regen ───────────────────────────────────────────────────
print(f'Shape regen (ds={args.ds}, cfg={args.shape_cfg})...')
t1 = time.time()

lc = torch.cat([torch.zeros(in_aabb.sum(), 1, dtype=torch.int32, device='cuda'), xyz[in_aabb]], 1)
x0 = (shape_slat.feats[in_aabb] - shape_mean) / shape_std
fm = pipeline.models['shape_slat_flow_model_1024']
C = fm.in_channels
if x0.shape[1] < C:
    x0 = torch.cat([x0, torch.zeros(x0.shape[0], C - x0.shape[1], device='cuda')], 1)

torch.manual_seed(args.seed)
eps = torch.randn_like(x0)
ni = (1 - args.ds) * x0 + args.ds * eps if args.ds < 1.0 else eps
start_t = args.ds if args.ds < 1.0 else 1.0

sv = None
if coarse_grid > 64:
    sv = pipeline._apply_ntk_rope_scaling(fm, coarse_grid / 64)
try:
    fm.to('cuda')
    ls = pipeline.shape_slat_sampler.sample(
        fm, SparseTensor(feats=ni, coords=lc), **cond,
        steps=args.steps, guidance_strength=args.shape_cfg,
        guidance_rescale=0.5, guidance_interval=[0.6, 1.0], rescale_t=3.0,
        start_t=start_t, verbose=True, tqdm_desc="Shape regen",
    ).samples
    fm.cpu()
    ls = ls * shape_std + shape_mean
finally:
    if sv:
        pipeline._restore_rope_freqs(fm, sv)

# ── Texture regen ─────────────────────────────────────────────────
if not args.shape_only:
    print(f'Texture regen (cfg={args.tex_cfg})...')
    sn = ls.replace(feats=(ls.feats - shape_mean) / shape_std)
    tfm = pipeline.models['tex_slat_flow_model_1024']

    tsv = None
    if coarse_grid > 64:
        tsv = pipeline._apply_ntk_rope_scaling(tfm, coarse_grid / 64)
    try:
        torch.manual_seed(args.seed + 1)
        tnc = tfm.in_channels - sn.feats.shape[1]
        tn = sn.replace(feats=torch.randn(sn.coords.shape[0], tnc).cuda())

        tfm.to('cuda')
        lt = pipeline.tex_slat_sampler.sample(
            tfm, tn, concat_cond=sn, **cond,
            steps=args.steps, guidance_strength=args.tex_cfg,
            guidance_rescale=0.0, guidance_interval=[0.6, 0.9], rescale_t=3.0,
            verbose=True, tqdm_desc="Texture regen",
        ).samples
        tfm.cpu()
        lt = lt * tex_std + tex_mean
    finally:
        if tsv:
            pipeline._restore_rope_freqs(tfm, tsv)
else:
    lt = None
    print('Texture: keeping original (--shape-only)')

# ── Feather blend + stitch (mask-distance based) ─────────────────
print(f'Stitching (feather={args.feather}, mask-based)...')
# Compute per-voxel min L-inf distance to any mask voxel
# mask=True voxels get distance 0, others get distance to nearest mask voxel
mask_coords = xyz[mask].float()  # [M, 3]
xyz_f = xyz.float()              # [N, 3]

# Chunked distance computation to avoid OOM on large tensors
chunk_size = 4096
min_dists = torch.full((xyz_f.shape[0],), float('inf'), device='cuda')
for i in range(0, mask_coords.shape[0], chunk_size):
    mc = mask_coords[i:i+chunk_size]  # [chunk, 3]
    # L-inf distance: max of abs diff per axis
    diffs = (xyz_f.unsqueeze(1) - mc.unsqueeze(0)).abs()  # [N, chunk, 3]
    linf = diffs.max(dim=2).values.min(dim=1).values      # [N]
    min_dists = torch.min(min_dists, linf)

# blend_w: 1.0 at mask, fades to 0.0 at feather distance
blend_w = (1.0 - (min_dists / args.feather).clamp(0, 1)).unsqueeze(1)  # [N, 1]
blend_w[~in_aabb] = 0.0  # only apply within AABB (where we have regen features)

sf = shape_slat.feats.clone()
sf[in_aabb] = (1 - blend_w[in_aabb]) * sf[in_aabb] + blend_w[in_aabb] * ls.feats

tf = tex_slat.feats.clone()
if lt is not None:
    tf[in_aabb] = (1 - blend_w[in_aabb]) * tf[in_aabb] + blend_w[in_aabb] * lt.feats

n_full = (blend_w[in_aabb] >= 1.0).all(dim=1).sum().item()
n_blend = in_aabb.sum().item() - n_full
print(f'  {n_full} full + {n_blend} feathered voxels')

# ── Decode + export ───────────────────────────────────────────────
print('Decoding...')
torch.cuda.empty_cache()
mesh = pipeline.decode_latent(
    shape_slat.replace(feats=sf), tex_slat.replace(feats=tf), res
)[0]
t2 = time.time()
print(f'  Mesh: {mesh.vertices.shape[0]:,} verts ({t2 - t1:.1f}s)')

os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
mesh.simplify(16777216)
vid = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave(f'{args.output}.mp4', vid, fps=15)

glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces,
    attr_volume=mesh.attrs, coords=mesh.coords,
    attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000, texture_size=4096,
    remesh=True, remesh_band=1, remesh_project=0, verbose=True,
)
glb.export(f'{args.output}.glb')
print(f'Done! → {args.output}.mp4 + .glb')
