"""Zero out masked features to visualize mask coverage."""
import os, sys, argparse
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, '/workspace')

import trimesh, numpy as np, torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor
import o_voxel

parser = argparse.ArgumentParser()
parser.add_argument('--latents', required=True)
parser.add_argument('--mask', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--mask-radius', type=int, default=1)
parser.add_argument('--padding', type=int, default=3)
args = parser.parse_args()

pipeline = Trellis2ImageTo3DPipeline.from_pretrained('/workspace/checkpoints/TRELLIS.2-4B')
pipeline.cuda()

data = torch.load(args.latents, map_location='cuda', weights_only=True)
shape_slat = SparseTensor(feats=data['shape_slat_feats'].cuda(), coords=data['shape_slat_coords'].cuda())
tex_slat = SparseTensor(feats=data['tex_slat_feats'].cuda(), coords=data['tex_slat_coords'].cuda())
res = data['res']
coarse_grid = res // 16

mask_mesh = trimesh.load(args.mask, force='mesh')
mask_verts = np.array(mask_mesh.vertices, dtype=np.float32)

for r in [args.mask_radius]:
    mask = pipeline.create_voxel_mask_from_mesh(mask_verts, shape_slat.coords, res, radius=r)
    n = mask.sum().item()
    print(f'radius={r}: {n}/{mask.shape[0]} voxels ({100*n/mask.shape[0]:.1f}%)')

    # Also show AABB with padding
    masked_xyz = shape_slat.coords[mask][:, 1:]
    aabb_min = (masked_xyz.min(0).values - args.padding).clamp(min=0)
    aabb_max = (masked_xyz.max(0).values + args.padding).clamp(max=coarse_grid-1)
    xyz = shape_slat.coords[:, 1:]
    in_aabb = ((xyz[:,0]>=aabb_min[0])&(xyz[:,0]<=aabb_max[0])&
               (xyz[:,1]>=aabb_min[1])&(xyz[:,1]<=aabb_max[1])&
               (xyz[:,2]>=aabb_min[2])&(xyz[:,2]<=aabb_max[2]))
    print(f'  AABB (padding={args.padding}): {in_aabb.sum().item()} voxels')
    print(f'  AABB range: [{aabb_min.tolist()}] → [{aabb_max.tolist()}]')

    # Zero out masked shape features and decode
    zeroed = shape_slat.feats.clone()
    zeroed[mask] = 0.0
    torch.cuda.empty_cache()
    mesh = pipeline.decode_latent(shape_slat.replace(feats=zeroed), tex_slat, res)[0]
    mesh.simplify(16777216)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces,
        attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
        decimation_target=250000, texture_size=2048,
        remesh=True, remesh_band=1, remesh_project=0, verbose=True)
    glb.export(f'{args.output}.glb')
    print(f'Saved: {args.output}.glb')
