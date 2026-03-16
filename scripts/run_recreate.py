"""
Recreate: regenerate sparse structure (Stage 0) in masked region,
then fully regenerate shape & texture on the merged coordinates.

Loads pre-saved latents (including z_s) — no re-generation needed.

Usage:
  # Single image:
  python scripts/run_recreate.py \
    --latents outputs/promote/latents.pt \
    --mask assets/promote/regen/mask_right.glb \
    --new-image "assets/promote/regen images/right-plus/right-plus1.png" \
    --output outputs/recreate/right-plus1 \
    --recreate-seed 789

  # Batch (multiple new-images):
  python scripts/run_recreate.py \
    --latents outputs/promote/latents.pt \
    --mask assets/promote/regen/mask_right.glb \
    --new-image "assets/promote/regen images/right-plus/right-plus1.png" \
                "assets/promote/regen images/right-plus/right-plus2.png" \
    --output outputs/recreate/promote_right \
    --recreate-seed 789
"""
import os, sys, argparse, time
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, '/workspace')

import cv2, trimesh, imageio
import numpy as np, torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

parser = argparse.ArgumentParser()
parser.add_argument('--latents', required=True, help='Path to latents.pt (must include z_s)')
parser.add_argument('--mask', required=True, help='GLB mask mesh path')
parser.add_argument('--new-image', nargs='+', required=True,
                    help='Conditioning image(s). If multiple, runs one recreate per image.')
parser.add_argument('--output', required=True, help='Output directory')
parser.add_argument('--recreate-seed', type=int, default=789, help='Seed for recreated region')
parser.add_argument('--mask-radius', type=int, default=1, help='Mask dilation radius (decoded-res voxels)')
parser.add_argument('--pipeline-type', type=str, default='1536_cascade', help='Pipeline type')
parser.add_argument('--ds', type=float, default=1.0, help='Denoise strength for cascade')
parser.add_argument('--shape-cfg', type=float, default=7.5, help='Shape guidance strength')
parser.add_argument('--tex-cfg', type=float, default=1.0, help='Texture guidance strength')
parser.add_argument('--feather', type=int, default=2, help='Feather blending width in voxels')
parser.add_argument('--ss-start-t', type=float, default=0.5,
                    help='Stage 0 SDEdit start_t (0=keep original, 1=pure noise)')
args = parser.parse_args()

CHECKPOINT = '/workspace/checkpoints/TRELLIS.2-4B'

pipeline = Trellis2ImageTo3DPipeline.from_pretrained(CHECKPOINT)
pipeline.cuda()
print('Pipeline loaded')

envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('/workspace/assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# Load latents
print(f'Loading latents: {args.latents}')
data = torch.load(args.latents, map_location='cuda', weights_only=True)
z_s = data['z_s'].cuda()
shape_slat = SparseTensor(feats=data['shape_slat_feats'].cuda(), coords=data['shape_slat_coords'].cuda())
tex_slat = SparseTensor(feats=data['tex_slat_feats'].cuda(), coords=data['tex_slat_coords'].cuda())
res = data['res']
coarse_grid = res // 16
print(f'  res={res}, grid={coarse_grid}³, voxels={shape_slat.coords.shape[0]}, z_s={list(z_s.shape)}')

# Load mask
mask_mesh = trimesh.load(args.mask, force='mesh')
mask_verts = np.array(mask_mesh.vertices, dtype=np.float32)

os.makedirs(args.output, exist_ok=True)

# Run recreate for each new-image
for img_path in args.new_image:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f'\n{"="*60}')
    print(f'Recreate: {name}')
    print(f'{"="*60}')

    new_image = Image.open(img_path)

    t1 = time.time()
    meshes = pipeline.run_recreate(
        image=new_image,
        mask_vertices=mask_verts,
        z_s=z_s,
        shape_slat=shape_slat,
        tex_slat=tex_slat,
        res=res,
        recreate_seed=args.recreate_seed,
        pipeline_type=args.pipeline_type,
        denoise_strength=args.ds,
        mask_radius=args.mask_radius,
        feather=args.feather,
        ss_start_t=args.ss_start_t,
        shape_slat_sampler_params={
            'guidance_strength': args.shape_cfg,
        },
        tex_slat_sampler_params={
            'guidance_strength': args.tex_cfg,
        },
    )
    t2 = time.time()
    print(f'Recreate done in {t2-t1:.1f}s')

    # Export mesh
    mesh = meshes[0]
    mesh.simplify(16777216)
    out_path = os.path.join(args.output, name)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], decimation_target=500000,
        texture_size=2048, remesh=True, remesh_band=1, remesh_project=0, verbose=True)
    glb.export(f'{out_path}.glb')
    print(f'Saved: {out_path}.glb')

    # Render preview
    vid = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
    imageio.mimsave(f'{out_path}.mp4', vid, fps=15)
    print(f'Saved: {out_path}.mp4')

    torch.cuda.empty_cache()

print(f'\nAll done! Results in: {args.output}')
