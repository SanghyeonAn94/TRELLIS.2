"""Regenerate masked region as standalone — no remap, no inpainting."""
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

CHECKPOINT = '/workspace/checkpoints/TRELLIS.2-4B'
INPUT_IMAGE = '/workspace/assets/inhouse/man.png'
MASK_PATH = '/workspace/assets/masks/man_mask_head.glb'
OUTPUT_DIR = '/workspace/outputs/local_regen'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
REGEN_SEED = 456
DENOISE_STRENGTH = 0.8
REGEN_CFG = 1.0
PADDING = 3
MASK_RADIUS = 1
STEPS = 20

ORIG_QUALITY = dict(
    sparse_structure_sampler_params={'steps': STEPS, 'guidance_strength': 3.0, 'guidance_rescale': 0.7, 'guidance_interval': [0.6, 1.0], 'rescale_t': 5.0},
    shape_slat_sampler_params={'steps': STEPS, 'guidance_strength': 7.5, 'guidance_rescale': 0.5, 'guidance_interval': [0.6, 1.0], 'rescale_t': 3.0},
    tex_slat_sampler_params={'steps': STEPS, 'guidance_strength': 1.0, 'guidance_rescale': 0.0, 'guidance_interval': [0.6, 0.9], 'rescale_t': 3.0},
)

pipeline = Trellis2ImageTo3DPipeline.from_pretrained(CHECKPOINT)
pipeline.cuda()

envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('/workspace/assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

image = Image.open(INPUT_IMAGE)
preprocessed = pipeline.preprocess_image(image)

# 1. Generate original
print("Step 1: Generating original...")
_, (shape_slat, tex_slat, res, _z_s) = pipeline.run(
    preprocessed, seed=SEED, return_latent=True, preprocess_image=False,
    pipeline_type='1024_1536', denoise_strength=1.0, max_num_tokens=131072, **ORIG_QUALITY,
)
coarse_grid = res // 16
print(f"  res={res}, grid={coarse_grid}³, voxels={shape_slat.coords.shape[0]}")

# 2. Extract AABB
print("Step 2: Extracting AABB...")
mask_mesh = trimesh.load(MASK_PATH, force='mesh')
mask_verts = np.array(mask_mesh.vertices, dtype=np.float32)
mask = pipeline.create_voxel_mask_from_mesh(mask_verts, shape_slat.coords, res, radius=MASK_RADIUS)

masked_xyz = shape_slat.coords[mask][:, 1:]
aabb_min = masked_xyz.min(dim=0).values - PADDING
aabb_max = masked_xyz.max(dim=0).values + PADDING
aabb_min = aabb_min.clamp(min=0)
aabb_max = aabb_max.clamp(max=coarse_grid - 1)
print(f"  AABB: [{aabb_min.tolist()}] → [{aabb_max.tolist()}]")

# Extract AABB voxels — ORIGINAL coords, no remap
coords_xyz = shape_slat.coords[:, 1:]
in_aabb = (
    (coords_xyz[:, 0] >= aabb_min[0]) & (coords_xyz[:, 0] <= aabb_max[0]) &
    (coords_xyz[:, 1] >= aabb_min[1]) & (coords_xyz[:, 1] <= aabb_max[1]) &
    (coords_xyz[:, 2] >= aabb_min[2]) & (coords_xyz[:, 2] <= aabb_max[2])
)
local_feats = shape_slat.feats[in_aabb]
local_coords = torch.cat([
    torch.zeros(in_aabb.sum().item(), 1, dtype=torch.int32, device='cuda'),
    coords_xyz[in_aabb],
], dim=1)
print(f"  AABB voxels: {local_coords.shape[0]}")

# Also extract texture
local_tex_feats = tex_slat.feats[in_aabb]

# 3. Shape: partial denoise from coarse features
print("Step 3: Shape regen (partial denoise)...")
shape_std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].cuda()
shape_mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].cuda()

flow_model = pipeline.models['shape_slat_flow_model_1024']
x0 = (local_feats - shape_mean) / shape_std

torch.manual_seed(REGEN_SEED)
C = flow_model.in_channels
if x0.shape[1] < C:
    x0 = torch.cat([x0, torch.zeros(x0.shape[0], C - x0.shape[1], device='cuda')], dim=1)

t = DENOISE_STRENGTH
eps = torch.randn_like(x0)
noisy_input = (1 - t) * x0 + t * eps

noise = SparseTensor(feats=noisy_input, coords=local_coords)
cond = pipeline.get_cond([preprocessed], 1024)

# NTK RoPE if needed
model_native = 64
saved = None
if coarse_grid > model_native:
    saved = pipeline._apply_ntk_rope_scaling(flow_model, coarse_grid / model_native)

try:
    flow_model.to('cuda')
    local_shape = pipeline.shape_slat_sampler.sample(
        flow_model, noise,
        **cond,
        steps=STEPS, guidance_strength=REGEN_CFG,
        guidance_rescale=0.5, guidance_interval=[0.6, 1.0], rescale_t=3.0,
        start_t=t, verbose=True, tqdm_desc="Regen shape",
    ).samples
    flow_model.cpu()
    local_shape = local_shape * shape_std + shape_mean
finally:
    if saved:
        pipeline._restore_rope_freqs(flow_model, saved)

# 4. Texture: full denoise conditioned on new shape
print("Step 4: Texture regen...")
tex_std = torch.tensor(pipeline.tex_slat_normalization['std'])[None].cuda()
tex_mean = torch.tensor(pipeline.tex_slat_normalization['mean'])[None].cuda()

shape_norm = local_shape.replace(feats=(local_shape.feats - shape_mean) / shape_std)
tex_flow_model = pipeline.models['tex_slat_flow_model_1024']

tex_saved = None
if coarse_grid > model_native:
    tex_saved = pipeline._apply_ntk_rope_scaling(tex_flow_model, coarse_grid / model_native)

try:
    torch.manual_seed(REGEN_SEED + 1)
    tex_in_ch = tex_flow_model.in_channels
    tex_noise_ch = tex_in_ch - shape_norm.feats.shape[1]
    tex_noise = shape_norm.replace(feats=torch.randn(shape_norm.coords.shape[0], tex_noise_ch).cuda())

    tex_flow_model.to('cuda')
    local_tex = pipeline.tex_slat_sampler.sample(
        tex_flow_model, tex_noise,
        concat_cond=shape_norm,
        **cond,
        steps=STEPS, guidance_strength=1.0,
        guidance_rescale=0.0, guidance_interval=[0.6, 0.9], rescale_t=3.0,
        verbose=True, tqdm_desc="Regen texture",
    ).samples
    tex_flow_model.cpu()
    local_tex = local_tex * tex_std + tex_mean
finally:
    if tex_saved:
        pipeline._restore_rope_freqs(tex_flow_model, tex_saved)

# 5. Decode
print("Step 5: Decoding...")
torch.cuda.empty_cache()
local_meshes = pipeline.decode_latent(local_shape, local_tex, res)
mesh = local_meshes[0]
print(f"  Local mesh: {mesh.vertices.shape[0]:,} vertices")

# 6. Export
print("Step 6: Exporting...")
mesh.simplify(16777216)
vid = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave(f'{OUTPUT_DIR}/regen_head_ds{DENOISE_STRENGTH}.mp4', vid, fps=15)

glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces,
    attr_volume=mesh.attrs, coords=mesh.coords,
    attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000, texture_size=4096,
    remesh=True, remesh_band=1, remesh_project=0, verbose=True,
)
glb.export(f'{OUTPUT_DIR}/regen_head_ds{DENOISE_STRENGTH}.glb')
print(f"Done! → {OUTPUT_DIR}/regen_head.mp4 + .glb")
