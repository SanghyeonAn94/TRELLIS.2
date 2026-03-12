"""Regen body with 4 different images — GPU 5"""
import subprocess, sys
LATENTS = '/workspace/assets/promote/regen/latents.pt'
MASK = '/workspace/assets/promote/regen/mask_body.glb'
IMAGES = [
    '/workspace/assets/promote/inpaint_images/body/body-1.png',
    '/workspace/assets/promote/inpaint_images/body/body-2.png',
    '/workspace/assets/promote/inpaint_images/body/body-3.png',
    '/workspace/assets/promote/inpaint_images/body/body-4.png',
]
for i, img in enumerate(IMAGES, 1):
    print(f'\n{"="*60}\nBody {i}: {img}\n{"="*60}')
    subprocess.run([sys.executable, 'scripts/run_regen_with_image.py',
        '--latents', LATENTS, '--mask', MASK, '--new-image', img,
        '--output', f'/workspace/outputs/promote/regen_body{i}',
        '--ds', '0.8', '--shape-cfg', '3.0', '--tex-cfg', '3.0', '--feather', '2',
    ], check=True)
print('\nAll body regen done!')
