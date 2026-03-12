"""Generate 3D model from input image using 2048 cascade."""
import os, sys, time
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, '/workspace')

import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

CHECKPOINT = '/workspace/checkpoints/TRELLIS.2-4B'
INPUT_IMAGE = '/workspace/assets/promote/input_image/2026-03-11_12-47-52_inpaint_1.png'
OUTPUT_DIR = '/workspace/outputs/promote'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
STEPS = 20

pipeline = Trellis2ImageTo3DPipeline.from_pretrained(CHECKPOINT)
pipeline.cuda()
print(f'Pipeline loaded')

image = Image.open(INPUT_IMAGE)
print(f'Input: {INPUT_IMAGE} ({image.size})')

t1 = time.time()
results = pipeline.run(
    image, seed=SEED, return_latent=True,
    pipeline_type='1536_cascade',
    sparse_structure_sampler_params={
        'steps': STEPS, 'guidance_strength': 7.5,
        'guidance_rescale': 0.7, 'guidance_interval': [0.6, 1.0], 'rescale_t': 5.0,
    },
    shape_slat_sampler_params={
        'steps': STEPS, 'guidance_strength': 7.5,
        'guidance_rescale': 0.5, 'guidance_interval': [0.6, 1.0], 'rescale_t': 3.0,
    },
    tex_slat_sampler_params={
        'steps': STEPS, 'guidance_strength': 1.0,
        'guidance_rescale': 0.0, 'guidance_interval': [0.6, 0.9], 'rescale_t': 3.0,
    },
    denoise_strength=1.0,
    max_num_tokens=131072,
)
meshes, (shape_slat, tex_slat, res) = results
t2 = time.time()
print(f'Generation done: {t2-t1:.1f}s, res={res}, voxels={shape_slat.coords.shape[0]}')

# Save latents for future regen/stitch
torch.save({
    'shape_slat_feats': shape_slat.feats.cpu(),
    'shape_slat_coords': shape_slat.coords.cpu(),
    'tex_slat_feats': tex_slat.feats.cpu(),
    'tex_slat_coords': tex_slat.coords.cpu(),
    'res': res,
    'seed': SEED,
}, f'{OUTPUT_DIR}/latents.pt')
print(f'Latents saved: {OUTPUT_DIR}/latents.pt')

# Export GLB only (half quality for speed)
mesh = meshes[0]
mesh.simplify(16777216)

glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces,
    attr_volume=mesh.attrs, coords=mesh.coords,
    attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=500000, texture_size=4096,
    remesh=True, remesh_band=1, remesh_project=0, verbose=True,
)
glb.export(f'{OUTPUT_DIR}/model_2048.glb')

t3 = time.time()
print(f'Export done: {t3-t2:.1f}s | Total: {t3-t1:.1f}s')
print(f'Results: {OUTPUT_DIR}/model_2048.glb')
