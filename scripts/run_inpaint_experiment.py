"""
RePaint-style 3D Inpainting Experiment
=======================================
Regenerates masked regions of a 3D model while preserving the rest.

Usage (inside container):
    python run_inpaint_experiment.py

Prerequisites:
    - Mask GLBs in assets/masks/
    - Input image in assets/inhouse/man.png
"""
import os, sys, time
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, '/workspace')

import cv2
import imageio
import trimesh
import numpy as np
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# ── Configuration ─────────────────────────────────────────────────
INPUT_IMAGE = '/workspace/assets/inhouse/man.png'
MASK_DIR = '/workspace/assets/masks'
OUTPUT_DIR = '/workspace/outputs/inpaint'
CHECKPOINT = '/workspace/checkpoints/TRELLIS.2-4B'

PIPELINE_TYPE = '1024_1536'
SEED = 42
INPAINT_SEED = 123
MAX_NUM_TOKENS = 131072
STEPS = 20
MASK_RADIUS = 5

MASKS = {
    'head':  os.path.join(MASK_DIR, 'man_mask_head.glb'),
    'chest': os.path.join(MASK_DIR, 'man_mask_chest.glb'),
}

# Original generation params (fixed — must match the original man model)
ORIG_QUALITY = dict(
    sparse_structure_sampler_params={
        'steps': STEPS,
        'guidance_strength': 3.0,
        'guidance_rescale': 0.7,
        'guidance_interval': [0.6, 1.0],
        'rescale_t': 5.0,
    },
    shape_slat_sampler_params={
        'steps': STEPS,
        'guidance_strength': 7.5,
        'guidance_rescale': 0.5,
        'guidance_interval': [0.6, 1.0],
        'rescale_t': 3.0,
    },
    tex_slat_sampler_params={
        'steps': STEPS,
        'guidance_strength': 1.0,
        'guidance_rescale': 0.0,
        'guidance_interval': [0.6, 0.9],
        'rescale_t': 3.0,
    },
)

# Inpaint pass params (low CFG for visible variation)
INPAINT_SHAPE_PARAMS = {
    'steps': STEPS,
    'guidance_strength': 1.0,
    'guidance_rescale': 0.5,
    'guidance_interval': [0.6, 1.0],
    'rescale_t': 3.0,
}

# ── Setup ─────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Loading pipeline...')
t0 = time.time()
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(CHECKPOINT)
pipeline.cuda()
print(f'Pipeline loaded: {time.time()-t0:.1f}s')

envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('/workspace/assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

image = Image.open(INPUT_IMAGE)
image.load()
print(f'Input image: {INPUT_IMAGE} ({image.size})')


def export_mesh(mesh, name):
    """Render video and export GLB for a mesh."""
    mesh.simplify(16777216)

    # Video
    video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
    mp4_path = os.path.join(OUTPUT_DIR, f'{name}.mp4')
    imageio.mimsave(mp4_path, video, fps=15)
    print(f'  Video: {mp4_path}')

    # GLB
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True
    )
    glb_path = os.path.join(OUTPUT_DIR, f'{name}.glb')
    glb.export(glb_path)
    print(f'  GLB: {glb_path}')


# ── Run inpainting for each mask ──────────────────────────────────
for mask_name, mask_path in MASKS.items():
    print(f'\n{"="*60}')
    print(f'Inpainting: {mask_name}')
    print(f'{"="*60}')

    # Extract vertices from mask GLB
    mask_mesh = trimesh.load(mask_path, force='mesh')
    mask_vertices = np.array(mask_mesh.vertices, dtype=np.float32)
    print(f'Mask mesh: {len(mask_vertices)} vertices')

    t1 = time.time()
    meshes = pipeline.run_inpaint(
        image,
        mask_vertices,
        seed=SEED,
        inpaint_seed=INPAINT_SEED,
        pipeline_type=PIPELINE_TYPE,
        denoise_strength=1.0,
        max_num_tokens=MAX_NUM_TOKENS,
        mask_radius=MASK_RADIUS,
        **ORIG_QUALITY,
        inpaint_shape_slat_sampler_params=INPAINT_SHAPE_PARAMS,
    )
    t2 = time.time()
    print(f'Inpainting done: {t2-t1:.1f}s')

    export_mesh(meshes[0], f'man_inpaint_{mask_name}')
    t3 = time.time()
    print(f'Export done: {t3-t2:.1f}s | Total: {t3-t1:.1f}s')

print(f'\nAll done! Results in {OUTPUT_DIR}')
