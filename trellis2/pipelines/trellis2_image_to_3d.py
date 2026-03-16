from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        return_latent: bool = False,
        inpaint_mask: Optional[torch.Tensor] = None,
        inpaint_x0: Optional[torch.Tensor] = None,
        inpaint_noise: Optional[torch.Tensor] = None,
        start_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            return_latent (bool): Whether to also return z_s latent.
            inpaint_mask (Optional[torch.Tensor]): [B, 1, R, R, R] bool mask for
                Stage 0 inpainting. True = regenerate, False = keep original.
            inpaint_x0 (Optional[torch.Tensor]): [B, C, R, R, R] original z_s latent
                for the known (unmasked) region.
            inpaint_noise (Optional[torch.Tensor]): [B, C, R, R, R] fixed noise for
                forward-noising the known region.
            start_noise (Optional[torch.Tensor]): Custom initial noise (e.g., noised z_s
                for SDEdit-style partial denoise). If None, uses random noise.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        if start_noise is not None:
            noise = start_noise
        else:
            noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            inpaint_mask=inpaint_mask,
            inpaint_x0=inpaint_x0,
            inpaint_noise=inpaint_noise,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        if return_latent:
            return coords, z_s
        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        hr_rope_scale: float = None,
        denoise_strength: float = 1.0,
    ) -> SparseTensor:
        """
        2-stage cascade: LR → HR with optional partial denoise and NTK RoPE scaling.

        Args:
            lr_cond (dict): Conditioning for the LR model.
            cond (dict): Conditioning for the HR model.
            flow_model_lr: The LR flow model.
            flow_model: The HR flow model.
            lr_resolution (int): LR output resolution (e.g. 512 or 1024).
            resolution (int): Target HR resolution (e.g. 1024, 1536, 2048).
            coords (torch.Tensor): Sparse structure coordinates.
            sampler_params (dict): Sampler params.
            max_num_tokens (int): Maximum token count for HR stage.
            hr_rope_scale (float): If set, apply NTK RoPE scaling for HR stage.
            denoise_strength (float): 1.0=full denoise, <1.0=partial denoise from upsampled features.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Stage 1: Sampling shape SLat (LR)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        lr_grid = lr_resolution // 16
        while True:
            hr_grid = hr_resolution // 16
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * hr_grid).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128

        hr_grid = hr_resolution // 16

        # Apply NTK RoPE scaling for HR stage if needed
        saved_freqs = None
        if hr_rope_scale is not None and hr_rope_scale > 1.0:
            saved_freqs = self._apply_ntk_rope_scaling(flow_model, hr_rope_scale)
            print(f"  NTK RoPE scaling: {lr_grid}³→{hr_grid}³ (scale={hr_rope_scale:.1f})")

        try:
            if denoise_strength >= 1.0:
                noisy_input = torch.randn(coords.shape[0], flow_model.in_channels).to(self.device)
                start_t = 1.0
            else:
                upsampled, found_mask = self._upsample_slat_features(
                    slat, coords, lr_grid, hr_grid)
                x0 = (upsampled.feats - mean) / std
                C_model = flow_model.in_channels
                C_feat = x0.shape[1]
                if C_feat < C_model:
                    x0 = torch.cat([x0, torch.zeros(x0.shape[0], C_model - C_feat, device=self.device)], dim=1)
                elif C_feat > C_model:
                    x0 = x0[:, :C_model]
                t = denoise_strength
                eps = torch.randn_like(x0)
                noisy_input = (1 - t) * x0 + t * eps
                noisy_input[~found_mask] = eps[~found_mask]
                start_t = denoise_strength

            print(f"  Stage 2: denoise_strength={denoise_strength:.2f}, start_t={start_t:.2f}")

            noise = SparseTensor(
                feats=noisy_input,
                coords=coords,
            )
            sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
            if self.low_vram:
                flow_model.to(self.device)
            slat = self.shape_slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                start_t=start_t,
                verbose=True,
                tqdm_desc="Stage 2: Sampling shape SLat (HR)",
            ).samples
            if self.low_vram:
                flow_model.cpu()

            std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
            mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
            slat = slat * std + mean

            return slat, hr_resolution
        finally:
            if saved_freqs is not None:
                self._restore_rope_freqs(flow_model, saved_freqs)

    def _apply_ntk_rope_scaling(self, model: nn.Module, scale_factor: float) -> dict:
        """
        Apply NTK-aware RoPE frequency scaling to all SparseRotaryPositionEmbedder modules.

        NTK-aware scaling preserves high-frequency components (local detail) while
        extending low-frequency components (global range) to handle larger grids.

        Formula: base_new = base_old * scale^(freq_dim / (freq_dim - 2))

        Args:
            model: The model whose RoPE modules to scale.
            scale_factor: The coordinate range extension factor (e.g. 2.0 for 64->128 grid).

        Returns:
            saved_freqs: Dict mapping module name to original freqs tensor for restoration.
        """
        from ..modules.sparse import SparseRotaryPositionEmbedder
        saved_freqs = {}
        for name, module in model.named_modules():
            if isinstance(module, SparseRotaryPositionEmbedder):
                saved_freqs[name] = module.freqs.clone()
                freq_dim = module.freq_dim
                old_base = module.rope_freq[1]
                new_base = old_base * (scale_factor ** (freq_dim / (freq_dim - 2)))
                idx = torch.arange(freq_dim, dtype=torch.float32, device=module.freqs.device) / freq_dim
                module.freqs = module.rope_freq[0] / (new_base ** idx)
        return saved_freqs

    def _restore_rope_freqs(self, model: nn.Module, saved_freqs: dict):
        """Restore original RoPE frequencies after NTK scaling."""
        from ..modules.sparse import SparseRotaryPositionEmbedder
        for name, module in model.named_modules():
            if isinstance(module, SparseRotaryPositionEmbedder):
                if name in saved_freqs:
                    module.freqs = saved_freqs[name]

    def _upsample_slat_features(
        self,
        coarse_slat: SparseTensor,
        fine_coords: torch.Tensor,
        coarse_grid: int,
        fine_grid: int,
    ) -> Tuple[SparseTensor, torch.Tensor]:
        """
        Map coarse SLat features to fine coordinates via nearest-neighbor lookup.

        Args:
            coarse_slat: The coarse structured latent.
            fine_coords: The fine coordinates (b, x, y, z) as int tensor.
            coarse_grid: The coarse grid resolution (e.g. 64).
            fine_grid: The fine grid resolution (e.g. 128).

        Returns:
            fine_slat: SparseTensor with features at fine_coords (zero where no parent).
            found_mask: Boolean tensor indicating which fine coords have a parent.
        """
        device = coarse_slat.device
        C = coarse_slat.feats.shape[1]

        # Build hash table for coarse coords -> feature index
        coarse_coords = coarse_slat.coords  # (N, 4): [batch, x, y, z]
        # Encode coarse coords as unique keys: batch * G^3 + x * G^2 + y * G + z
        coarse_keys = (
            coarse_coords[:, 0].long() * (coarse_grid ** 3)
            + coarse_coords[:, 1].long() * (coarse_grid ** 2)
            + coarse_coords[:, 2].long() * coarse_grid
            + coarse_coords[:, 3].long()
        )

        # Map fine coords to parent coarse coords (float division for non-integer ratios)
        parent_xyz = (fine_coords[:, 1:].float() * coarse_grid / fine_grid).long()
        parent_coords = torch.cat([
            fine_coords[:, :1],
            parent_xyz,
        ], dim=1)
        parent_keys = (
            parent_coords[:, 0].long() * (coarse_grid ** 3)
            + parent_coords[:, 1].long() * (coarse_grid ** 2)
            + parent_coords[:, 2].long() * coarse_grid
            + parent_coords[:, 3].long()
        )

        # Build lookup via hash table (using torch unique + searchsorted)
        unique_coarse_keys, inverse_coarse = torch.unique(coarse_keys, sorted=True, return_inverse=True)
        # For duplicate keys, take the last one (shouldn't have duplicates for valid sparse structure)
        key_to_idx = torch.empty(unique_coarse_keys.shape[0], dtype=torch.long, device=device)
        key_to_idx[inverse_coarse] = torch.arange(coarse_keys.shape[0], device=device)

        # Search parent keys in coarse keys
        search_pos = torch.searchsorted(unique_coarse_keys, parent_keys)
        search_pos = search_pos.clamp(0, unique_coarse_keys.shape[0] - 1)
        found_mask = unique_coarse_keys[search_pos] == parent_keys

        # Build fine features
        fine_feats = torch.zeros(fine_coords.shape[0], C, device=device, dtype=coarse_slat.feats.dtype)
        matched_coarse_idx = key_to_idx[search_pos[found_mask]]
        fine_feats[found_mask] = coarse_slat.feats[matched_coarse_idx]

        fine_slat = SparseTensor(
            feats=fine_feats,
            coords=fine_coords,
        )
        return fine_slat, found_mask

    def sample_shape_slat_3stage_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        mid_resolution: int,
        hr_resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        hr_sampler_params: dict = {},
        denoise_strength: float = 0.5,
        max_num_tokens: int = 49152,
    ) -> Tuple[SparseTensor, int]:
        """
        3-stage cascade: LR(512) -> MID(1024, full denoise) -> HR(2048, partial denoise).

        Args:
            lr_cond: Conditioning for the LR model.
            cond: Conditioning for the 1024 model (used for both stage 2 and 3).
            flow_model_lr: The 512 resolution flow model.
            flow_model: The 1024 resolution flow model (reused for stage 3).
            lr_resolution: LR output resolution (e.g. 512).
            mid_resolution: MID output resolution (e.g. 1024).
            hr_resolution: Target HR resolution (e.g. 2048).
            coords: Sparse structure coordinates from stage 0.
            sampler_params: Sampler params for stage 1 & 2.
            hr_sampler_params: Sampler params for stage 3 (overrides sampler_params).
            denoise_strength: Partial denoise strength for stage 3 (0=no change, 1=full denoise).
            max_num_tokens: Maximum token count for HR stage.

        Returns:
            slat: The final HR structured latent.
            final_hr_resolution: The actual HR resolution used (may be reduced if token limit hit).
        """
        # ============================================================
        # Stage 1: LR (512 model, full denoise) → slat_lr @ 32³
        # ============================================================
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params_merged = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat_lr = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params_merged,
            verbose=True,
            tqdm_desc="Stage 1: Sampling shape SLat (LR)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat_lr.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat_lr.device)
        slat_lr = slat_lr * std + mean

        # ============================================================
        # Upsample LR → MID coords
        # ============================================================
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        mid_hr_coords = self.models['shape_slat_decoder'].upsample(slat_lr, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False

        mid_grid = mid_resolution // 16  # 1024 // 16 = 64
        quant_coords = torch.cat([
            mid_hr_coords[:, :1],
            ((mid_hr_coords[:, 1:] + 0.5) / lr_resolution * mid_grid).int(),
        ], dim=1)
        mid_coords = quant_coords.unique(dim=0)

        # ============================================================
        # Stage 2: MID (1024 model, full denoise) → slat_mid @ 64³
        # ============================================================
        noise = SparseTensor(
            feats=torch.randn(mid_coords.shape[0], flow_model.in_channels).to(self.device),
            coords=mid_coords,
        )
        if self.low_vram:
            flow_model.to(self.device)
        slat_mid = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params_merged,
            verbose=True,
            tqdm_desc="Stage 2: Sampling shape SLat (MID)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        slat_mid = slat_mid * std + mean

        # ============================================================
        # Upsample MID → HR coords
        # ============================================================
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_raw_coords = self.models['shape_slat_decoder'].upsample(slat_mid, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False

        actual_hr_resolution = hr_resolution
        while True:
            hr_grid = actual_hr_resolution // 16  # 2048//16=128
            quant_hr_coords = torch.cat([
                hr_raw_coords[:, :1],
                ((hr_raw_coords[:, 1:] + 0.5) / mid_resolution * hr_grid).int(),
            ], dim=1)
            hr_coords = quant_hr_coords.unique(dim=0)
            num_tokens = hr_coords.shape[0]
            if num_tokens < max_num_tokens or actual_hr_resolution == 1024:
                if actual_hr_resolution != hr_resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {actual_hr_resolution}.")
                break
            actual_hr_resolution -= 128

        hr_grid = actual_hr_resolution // 16

        # ============================================================
        # Stage 3: HR (1024 model reused, partial or full denoise)
        # denoise_strength=1.0 → full denoise from pure noise
        # denoise_strength<1.0 → partial denoise from upsampled features
        # NTK-aware RoPE scaling applied for grid extrapolation.
        # ============================================================
        rope_scale = hr_grid / mid_grid  # e.g. 128/64 = 2.0
        saved_freqs = None
        if rope_scale > 1.0:
            saved_freqs = self._apply_ntk_rope_scaling(flow_model, rope_scale)
            print(f"  NTK RoPE scaling: {mid_grid}³→{hr_grid}³ (scale={rope_scale:.1f})")

        try:
            if denoise_strength >= 1.0:
                # Full denoise from pure noise
                noisy_input = torch.randn(hr_coords.shape[0], flow_model.in_channels).to(self.device)
                start_t = 1.0
            else:
                # Partial denoise: upsample Stage 2 features → add noise
                upsampled, found_mask = self._upsample_slat_features(
                    slat_mid, hr_coords, mid_grid, hr_grid)
                # Normalize upsampled features to model's expected distribution
                x0 = (upsampled.feats - mean) / std
                # Pad/truncate channels to match flow model input
                C_model = flow_model.in_channels
                C_feat = x0.shape[1]
                if C_feat < C_model:
                    x0 = torch.cat([x0, torch.zeros(x0.shape[0], C_model - C_feat, device=self.device)], dim=1)
                elif C_feat > C_model:
                    x0 = x0[:, :C_model]
                # Create noisy input: x_t = (1-t)*x0 + t*noise
                t = denoise_strength
                eps = torch.randn_like(x0)
                noisy_input = (1 - t) * x0 + t * eps
                # Orphan voxels (no parent) get pure noise
                noisy_input[~found_mask] = eps[~found_mask]
                start_t = denoise_strength

            print(f"  Stage 3: denoise_strength={denoise_strength:.2f}, start_t={start_t:.2f}")

            noise = SparseTensor(
                feats=noisy_input,
                coords=hr_coords,
            )
            hr_sampler_params_merged = {**self.shape_slat_sampler_params, **sampler_params, **hr_sampler_params}
            if self.low_vram:
                flow_model.to(self.device)
            slat_hr = self.shape_slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **hr_sampler_params_merged,
                start_t=start_t,
                verbose=True,
                tqdm_desc="Stage 3: Sampling shape SLat (HR)",
            ).samples
            if self.low_vram:
                flow_model.cpu()

            slat_hr = slat_hr * std + mean

            return slat_hr, actual_hr_resolution

        finally:
            if saved_freqs is not None:
                self._restore_rope_freqs(flow_model, saved_freqs)

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        denoise_strength: float = 0.5,
        hr_shape_slat_sampler_params: dict = {},
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade', '2048_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
            denoise_strength (float): Partial denoise strength for 2048_cascade stage 3 (0=no change, 1=full denoise).
            hr_shape_slat_sampler_params (dict): Sampler params for 2048_cascade stage 3 (overrides shape_slat_sampler_params).
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type in ('1024_cascade', '1536_cascade', '2048_cascade'):
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type in ('1024_1536', '1024_2048'):
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32, '2048_cascade': 32, '1024_1536': 64, '1024_2048': 64}[pipeline_type]
        ss_result = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params,
            return_latent=return_latent,
        )
        if return_latent:
            coords, z_s = ss_result
        else:
            coords = ss_result
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                denoise_strength=denoise_strength,
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                denoise_strength=denoise_strength,
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '2048_cascade':
            shape_slat, res = self.sample_shape_slat_3stage_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024, 2048,
                coords, shape_slat_sampler_params,
                hr_shape_slat_sampler_params,
                denoise_strength,
                max_num_tokens
            )
            # Apply NTK RoPE scaling to texture model for 128³ grid
            hr_grid = res // 16  # actual_hr_resolution // 16
            mid_grid = 1024 // 16  # 64
            tex_rope_scale = hr_grid / mid_grid
            tex_saved_freqs = None
            if tex_rope_scale > 1.0:
                tex_saved_freqs = self._apply_ntk_rope_scaling(
                    self.models['tex_slat_flow_model_1024'], tex_rope_scale)
                print(f"  NTK RoPE scaling (texture): {mid_grid}³→{hr_grid}³ (scale={tex_rope_scale:.1f})")
            try:
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params
                )
            finally:
                if tex_saved_freqs is not None:
                    self._restore_rope_freqs(
                        self.models['tex_slat_flow_model_1024'], tex_saved_freqs)
        elif pipeline_type == '1024_1536':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_1024, cond_1024,
                self.models['shape_slat_flow_model_1024'], self.models['shape_slat_flow_model_1024'],
                1024, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                denoise_strength=denoise_strength,
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1024_2048':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_1024, cond_1024,
                self.models['shape_slat_flow_model_1024'], self.models['shape_slat_flow_model_1024'],
                1024, 2048,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                hr_rope_scale=2.0,
                denoise_strength=denoise_strength,
            )
            # Apply NTK RoPE scaling to texture model for 128³ grid
            hr_grid = res // 16
            mid_grid = 1024 // 16
            tex_rope_scale = hr_grid / mid_grid
            tex_saved_freqs = None
            if tex_rope_scale > 1.0:
                tex_saved_freqs = self._apply_ntk_rope_scaling(
                    self.models['tex_slat_flow_model_1024'], tex_rope_scale)
                print(f"  NTK RoPE scaling (texture): {mid_grid}³→{hr_grid}³ (scale={tex_rope_scale:.1f})")
            try:
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params
                )
            finally:
                if tex_saved_freqs is not None:
                    self._restore_rope_freqs(
                        self.models['tex_slat_flow_model_1024'], tex_saved_freqs)
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res, z_s)
        else:
            return out_mesh

    @staticmethod
    def create_voxel_mask_from_mesh(
        mask_vertices: np.ndarray,
        slat_coords: torch.Tensor,
        resolution: int,
        radius: int = 1,
    ) -> torch.Tensor:
        """
        Convert mask mesh vertices to a boolean voxel mask aligned with SLat coordinates.

        Args:
            mask_vertices (np.ndarray): [M, 3] vertices in [-0.5, 0.5]^3.
            slat_coords (torch.Tensor): [N, 4] SLat coordinates (batch, x, y, z).
            resolution (int): Pipeline resolution (e.g. 1536).
            radius (int): Dilation radius in voxels.

        Returns:
            torch.Tensor: Boolean mask [N], True = inside mask (to be regenerated).
        """
        grid_size = resolution // 16

        # GLB uses Y-up right-handed; SLat grid uses Z-up right-handed.
        # Mapping: GLB (x, y, z) → SLat grid (x, -z, y)
        mask_vertices = np.column_stack([
            mask_vertices[:, 0],    # GLB X → SLat X
            -mask_vertices[:, 2],   # -GLB Z → SLat Y (negate for handedness)
            mask_vertices[:, 1],    # GLB Y → SLat Z
        ])

        # Convert vertices [-0.5, 0.5] → grid coordinates [0, grid_size)
        voxel_coords = (mask_vertices + 0.5) * grid_size
        voxel_ints = np.clip(np.round(voxel_coords).astype(int), 0, grid_size - 1)

        # Build 3D occupancy grid
        occupied = np.zeros((grid_size,) * 3, dtype=bool)
        occupied[voxel_ints[:, 0], voxel_ints[:, 1], voxel_ints[:, 2]] = True

        # Dilate by radius
        if radius > 0:
            padded = np.pad(occupied, radius, mode='constant', constant_values=False)
            dilated = np.zeros_like(occupied)
            for dx in range(2 * radius + 1):
                for dy in range(2 * radius + 1):
                    for dz in range(2 * radius + 1):
                        dilated |= padded[dx:dx+grid_size, dy:dy+grid_size, dz:dz+grid_size]
            occupied = dilated

        # Lookup SLat coordinates
        xyz = slat_coords[:, 1:].cpu().numpy().astype(int)
        valid = (xyz >= 0).all(axis=1) & (xyz < grid_size).all(axis=1)
        mask = np.zeros(len(xyz), dtype=bool)
        mask[valid] = occupied[xyz[valid, 0], xyz[valid, 1], xyz[valid, 2]]

        return torch.tensor(mask, dtype=torch.bool, device=slat_coords.device)

    @staticmethod
    def create_dense_structure_mask(
        mask_vertices: np.ndarray,
        latent_res: int = 16,
        decoded_res: int = 64,
        radius: int = 1,
        device: torch.device = torch.device('cuda'),
    ) -> torch.Tensor:
        """
        Create a dense 3D mask at sparse structure latent resolution for Stage 0 inpainting.

        Args:
            mask_vertices (np.ndarray): [M, 3] vertices in [-0.5, 0.5]^3 (GLB Y-up).
            latent_res (int): Latent resolution of SS flow model (default 16).
            decoded_res (int): Decoded occupancy resolution (default 64).
            radius (int): Dilation radius in decoded-resolution voxels.
            device (torch.device): Target device.

        Returns:
            torch.Tensor: [1, 1, latent_res, latent_res, latent_res] bool mask.
                True = regenerate, False = keep original.
        """
        # GLB Y-up → SLat Z-up coordinate transform
        mask_vertices = np.column_stack([
            mask_vertices[:, 0],
            -mask_vertices[:, 2],
            mask_vertices[:, 1],
        ])

        # Vertices [-0.5, 0.5] → decoded_res grid coordinates
        voxel_coords = (mask_vertices + 0.5) * decoded_res
        voxel_ints = np.clip(np.round(voxel_coords).astype(int), 0, decoded_res - 1)

        # Build occupancy at decoded resolution
        occupied = np.zeros((decoded_res,) * 3, dtype=bool)
        occupied[voxel_ints[:, 0], voxel_ints[:, 1], voxel_ints[:, 2]] = True

        # Dilate by radius at decoded resolution
        if radius > 0:
            padded = np.pad(occupied, radius, mode='constant', constant_values=False)
            dilated = np.zeros_like(occupied)
            for dx in range(2 * radius + 1):
                for dy in range(2 * radius + 1):
                    for dz in range(2 * radius + 1):
                        dilated |= padded[dx:dx+decoded_res, dy:dy+decoded_res, dz:dz+decoded_res]
            occupied = dilated

        # Downscale to latent resolution via max_pool3d
        # Any occupied voxel in a block marks the latent voxel as masked
        occupied_t = torch.tensor(occupied, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ratio = decoded_res // latent_res
        mask_latent = torch.nn.functional.max_pool3d(occupied_t, ratio, ratio, 0) > 0.5

        return mask_latent.to(device)

    @torch.no_grad()
    def run_inpaint(
        self,
        image: Image.Image,
        mask_vertices: np.ndarray,
        seed: int = 42,
        inpaint_seed: int = 123,
        pipeline_type: Optional[str] = None,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        denoise_strength: float = 1.0,
        max_num_tokens: int = 131072,
        mask_radius: int = 1,
        inpaint_shape_slat_sampler_params: Optional[dict] = None,
        inpaint_tex_slat_sampler_params: Optional[dict] = None,
    ) -> List[MeshWithVoxel]:
        """
        RePaint-style 3D inpainting: regenerate masked voxels while preserving the rest.

        At each denoising step, the known (unmasked) region is replaced with a
        forward-noised version of the original latent, so only the masked region
        receives new content from the flow model.

        Args:
            image: Input image prompt.
            mask_vertices: [M, 3] vertices of the mask mesh in [-0.5, 0.5]^3.
            seed: Seed for reproducing the original generation.
            inpaint_seed: Seed for new content in the masked region.
            pipeline_type: Pipeline type (default: self.default_pipeline_type).
            sparse_structure_sampler_params: Sparse structure sampler overrides (for original).
            shape_slat_sampler_params: Shape SLat sampler overrides (for original).
            tex_slat_sampler_params: Texture SLat sampler overrides (for original).
            denoise_strength: Denoise strength for the original cascade generation.
            max_num_tokens: Maximum number of tokens.
            mask_radius: Dilation radius for voxel mask (voxels).
            inpaint_shape_slat_sampler_params: Shape sampler overrides for inpaint pass only.
                If None, uses shape_slat_sampler_params.
            inpaint_tex_slat_sampler_params: Texture sampler overrides for inpaint pass only.
                If None, uses tex_slat_sampler_params.

        Returns:
            List[MeshWithVoxel]: The inpainted meshes.
        """
        pipeline_type = pipeline_type or self.default_pipeline_type

        # Inpaint params default to original params if not specified
        if inpaint_shape_slat_sampler_params is None:
            inpaint_shape_slat_sampler_params = shape_slat_sampler_params
        if inpaint_tex_slat_sampler_params is None:
            inpaint_tex_slat_sampler_params = tex_slat_sampler_params

        # Preprocess image once
        preprocessed = self.preprocess_image(image)

        # ── Step 1: Generate original with same seed ──────────────
        print("[Inpaint] Step 1/5: Generating original model...")
        results = self.run(
            preprocessed, seed=seed, return_latent=True,
            preprocess_image=False,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            shape_slat_sampler_params=shape_slat_sampler_params,
            tex_slat_sampler_params=tex_slat_sampler_params,
            denoise_strength=denoise_strength,
            max_num_tokens=max_num_tokens,
        )
        _, (shape_slat, tex_slat, res, _z_s) = results

        # ── Step 2: Create voxel mask ─────────────────────────────
        print("[Inpaint] Step 2/5: Creating voxel mask...")
        mask = self.create_voxel_mask_from_mesh(
            mask_vertices, shape_slat.coords, res, radius=mask_radius
        )
        n_masked = mask.sum().item()
        n_total = mask.shape[0]
        print(f"  Mask: {n_masked}/{n_total} voxels ({100*n_masked/n_total:.1f}%)")

        # Get conditioning for inpaint pass
        cond_res = 512 if pipeline_type == '512' else 1024
        cond = self.get_cond([preprocessed], cond_res)

        # Normalization constants
        shape_std = torch.tensor(self.shape_slat_normalization['std'])[None].to(self.device)
        shape_mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(self.device)
        tex_std = torch.tensor(self.tex_slat_normalization['std'])[None].to(self.device)
        tex_mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(self.device)

        # ── Step 3: Shape inpainting ──────────────────────────────
        print("[Inpaint] Step 3/5: Shape inpainting...")
        shape_x0 = (shape_slat.feats - shape_mean) / shape_std

        shape_flow_model = (
            self.models['shape_slat_flow_model_512'] if pipeline_type == '512'
            else self.models['shape_slat_flow_model_1024']
        )

        # RoPE scaling (match original pipeline behavior)
        grid_size = res // 16
        model_native_grid = 32 if pipeline_type == '512' else 64
        shape_saved_freqs = None
        if pipeline_type in ('2048_cascade', '1024_2048') and grid_size > model_native_grid:
            rope_scale = grid_size / model_native_grid
            shape_saved_freqs = self._apply_ntk_rope_scaling(shape_flow_model, rope_scale)
            print(f"  NTK RoPE scaling (shape): {model_native_grid}³→{grid_size}³ (scale={rope_scale:.1f})")

        try:
            torch.manual_seed(inpaint_seed)
            noise_feats = torch.randn(shape_slat.coords.shape[0], shape_flow_model.in_channels).to(self.device)
            inpaint_noise = torch.randn_like(shape_x0)
            noise = shape_slat.replace(feats=noise_feats)

            sampler_params = {**self.shape_slat_sampler_params, **inpaint_shape_slat_sampler_params}
            if self.low_vram:
                shape_flow_model.to(self.device)

            shape_slat_new = self.shape_slat_sampler.sample(
                shape_flow_model, noise,
                **cond, **sampler_params,
                inpaint_mask=mask,
                inpaint_x0=shape_x0,
                inpaint_noise=inpaint_noise,
                verbose=True,
                tqdm_desc="Inpainting shape SLat",
            ).samples

            if self.low_vram:
                shape_flow_model.cpu()

            shape_slat_new = shape_slat_new * shape_std + shape_mean
        finally:
            if shape_saved_freqs is not None:
                self._restore_rope_freqs(shape_flow_model, shape_saved_freqs)

        # ── Step 4: Texture inpainting ────────────────────────────
        print("[Inpaint] Step 4/5: Texture inpainting...")
        tex_x0 = (tex_slat.feats - tex_mean) / tex_std

        # Normalize inpainted shape as concat_cond for texture model
        shape_slat_norm = shape_slat_new.replace(
            feats=(shape_slat_new.feats - shape_mean) / shape_std
        )

        tex_flow_model = (
            self.models['tex_slat_flow_model_512'] if pipeline_type == '512'
            else self.models['tex_slat_flow_model_1024']
        )

        tex_saved_freqs = None
        if pipeline_type in ('2048_cascade', '1024_2048') and grid_size > model_native_grid:
            tex_rope_scale = grid_size / model_native_grid
            tex_saved_freqs = self._apply_ntk_rope_scaling(tex_flow_model, tex_rope_scale)
            print(f"  NTK RoPE scaling (texture): {model_native_grid}³→{grid_size}³ (scale={tex_rope_scale:.1f})")

        try:
            torch.manual_seed(inpaint_seed + 1)
            tex_in_channels = (
                tex_flow_model.in_channels if isinstance(tex_flow_model, nn.Module)
                else tex_flow_model[0].in_channels
            )
            tex_noise_channels = tex_in_channels - shape_slat_norm.feats.shape[1]
            tex_noise_feats = torch.randn(shape_slat_norm.coords.shape[0], tex_noise_channels).to(self.device)
            tex_inpaint_noise = torch.randn_like(tex_x0)
            tex_noise = shape_slat_norm.replace(feats=tex_noise_feats)

            tex_sampler_params = {**self.tex_slat_sampler_params, **inpaint_tex_slat_sampler_params}
            if self.low_vram:
                tex_flow_model.to(self.device)

            tex_slat_new = self.tex_slat_sampler.sample(
                tex_flow_model, tex_noise,
                concat_cond=shape_slat_norm,
                **cond, **tex_sampler_params,
                inpaint_mask=mask,
                inpaint_x0=tex_x0,
                inpaint_noise=tex_inpaint_noise,
                verbose=True,
                tqdm_desc="Inpainting texture SLat",
            ).samples

            if self.low_vram:
                tex_flow_model.cpu()

            tex_slat_new = tex_slat_new * tex_std + tex_mean
        finally:
            if tex_saved_freqs is not None:
                self._restore_rope_freqs(tex_flow_model, tex_saved_freqs)

        # ── Step 5: Decode ────────────────────────────────────────
        print("[Inpaint] Step 5/5: Decoding...")
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat_new, tex_slat_new, res)

        return out_mesh

    @torch.no_grad()
    def run_recreate(
        self,
        image: Image.Image,
        mask_vertices: np.ndarray,
        z_s: torch.Tensor,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        res: int,
        recreate_seed: int = 789,
        pipeline_type: Optional[str] = None,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        denoise_strength: float = 1.0,
        max_num_tokens: int = 131072,
        mask_radius: int = 1,
        feather: int = 2,
        recreate_ss_sampler_params: Optional[dict] = None,
        ss_start_t: float = 0.5,
    ) -> List[MeshWithVoxel]:
        """
        Recreate: regenerate sparse structure (Stage 0) in the masked region,
        then generate shape/texture on merged coords and stitch with original features.

        Unlike regen/repaint which only change features at existing voxel positions,
        recreate can add/remove voxels - enabling entirely new geometry.

        Args:
            image: Conditioning image for the recreated region.
            mask_vertices: [M, 3] vertices of the mask mesh in [-0.5, 0.5]^3.
            z_s: [B, C, R, R, R] original Stage 0 latent from generation.
            shape_slat: Original shape SLat (HR features from latents.pt).
            tex_slat: Original texture SLat (HR features from latents.pt).
            res: Resolution of the original generation (e.g., 1536).
            recreate_seed: Seed for new content in the masked region.
            pipeline_type: Pipeline type (default: self.default_pipeline_type).
            sparse_structure_sampler_params: SS sampler overrides for recreate.
            shape_slat_sampler_params: Shape SLat sampler overrides.
            tex_slat_sampler_params: Texture SLat sampler overrides.
            denoise_strength: Denoise strength for cascade generation.
            max_num_tokens: Maximum number of tokens.
            mask_radius: Dilation radius for mask (in decoded-resolution voxels).
            feather: Feather blending width in voxels at HR resolution.
            recreate_ss_sampler_params: SS sampler overrides for recreate pass.
                If None, uses sparse_structure_sampler_params.
            ss_start_t: Start timestep for Stage 0 SDEdit (0.0-1.0).
                Lower = closer to original structure, higher = more freedom.
                Default 0.5 preserves structure well while allowing conditioning.

        Returns:
            List[MeshWithVoxel]: The recreated meshes (full model with stitching).
        """
        pipeline_type = pipeline_type or self.default_pipeline_type

        if recreate_ss_sampler_params is None:
            recreate_ss_sampler_params = sparse_structure_sampler_params

        preprocessed = self.preprocess_image(image)
        z_s = z_s.to(self.device)

        # ── Step 1: Create dense 3D mask at latent resolution ─────
        print("[Recreate] Step 1/6: Creating dense structure mask...")
        flow_model = self.models['sparse_structure_flow_model']
        latent_res = flow_model.resolution  # 16
        decoder = self.models['sparse_structure_decoder']

        if self.low_vram:
            decoder.to(self.device)
        orig_decoded = decoder(z_s) > 0  # [B, 1, D, D, D]
        if self.low_vram:
            decoder.cpu()
        decoded_res = orig_decoded.shape[2]

        mask_latent = self.create_dense_structure_mask(
            mask_vertices, latent_res=latent_res, decoded_res=decoded_res,
            radius=mask_radius, device=self.device,
        )
        print(f"  Latent mask: {mask_latent.sum().item()}/{mask_latent.numel()} "
              f"({100*mask_latent.sum().item()/mask_latent.numel():.1f}%)")

        mask_decoded = self.create_dense_structure_mask(
            mask_vertices, latent_res=decoded_res, decoded_res=decoded_res,
            radius=mask_radius, device=self.device,
        )

        # ── Step 2: Stage 0 SDEdit + inpainting ─────────────────
        print(f"[Recreate] Step 2/6: Recreating sparse structure (ss_start_t={ss_start_t})...")
        cond_512 = self.get_cond([preprocessed], 512)
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32,
                  '2048_cascade': 32, '1024_1536': 64, '1024_2048': 64}[pipeline_type]

        torch.manual_seed(recreate_seed)
        inpaint_noise = torch.randn_like(z_s)

        # SDEdit: start from noised z_s instead of pure noise
        # This seeds the mask region with existing structure so the model
        # modifies rather than generates from scratch
        sigma_min = self.sparse_structure_sampler.sigma_min
        sigma_start = sigma_min + (1 - sigma_min) * ss_start_t
        start_noise = (1 - ss_start_t) * z_s + sigma_start * inpaint_noise

        ss_params = {**recreate_ss_sampler_params, 'start_t': ss_start_t}
        new_coords, new_z_s = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples=1,
            sampler_params=ss_params,
            return_latent=True,
            inpaint_mask=mask_latent,
            inpaint_x0=z_s,
            inpaint_noise=inpaint_noise,
            start_noise=start_noise,
        )

        if self.low_vram:
            decoder.to(self.device)
        new_decoded = decoder(new_z_s) > 0
        if self.low_vram:
            decoder.cpu()

        # ── Step 3: Merge occupancy ───────────────────────────────
        print("[Recreate] Step 3/6: Merging occupancy...")
        orig_in_mask = int((orig_decoded & mask_decoded).sum())
        new_in_mask = int((new_decoded & mask_decoded).sum())
        print(f"  Occupancy in mask: {orig_in_mask} (orig) → {new_in_mask} (new)")

        merged_decoded = torch.where(mask_decoded, new_decoded, orig_decoded)

        if ss_res != merged_decoded.shape[2]:
            ratio = merged_decoded.shape[2] // ss_res
            merged_decoded = torch.nn.functional.max_pool3d(
                merged_decoded.float(), ratio, ratio, 0
            ) > 0.5
        merged_coords = torch.argwhere(merged_decoded)[:, [0, 2, 3, 4]].int()
        print(f"  Merged coarse voxels: {merged_coords.shape[0]}")

        # ── Step 4: Stage 1 + Stage 2 on merged coords ───────────
        print("[Recreate] Step 4/6: Generating shape & texture on merged structure...")
        cond_1024 = self.get_cond([preprocessed], 1024) if pipeline_type != '512' else None
        torch.manual_seed(recreate_seed)

        if pipeline_type == '512':
            shape_slat_new = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                merged_coords, shape_slat_sampler_params
            )
            tex_slat_new = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat_new, tex_slat_sampler_params
            )
            new_res = 512
        elif pipeline_type == '1024':
            shape_slat_new = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                merged_coords, shape_slat_sampler_params
            )
            tex_slat_new = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat_new, tex_slat_sampler_params
            )
            new_res = 1024
        elif pipeline_type in ('1024_cascade', '1536_cascade'):
            target_res = {'1024_cascade': 1024, '1536_cascade': 1536}[pipeline_type]
            shape_slat_new, new_res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'],
                self.models['shape_slat_flow_model_1024'],
                512, target_res,
                merged_coords, shape_slat_sampler_params,
                max_num_tokens=max_num_tokens,
                hr_rope_scale=target_res / 1024 if target_res > 1024 else None,
                denoise_strength=denoise_strength,
            )
            tex_slat_new = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat_new, tex_slat_sampler_params
            )
        elif pipeline_type == '2048_cascade':
            shape_slat_new, new_res = self.sample_shape_slat_3stage_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'],
                self.models['shape_slat_flow_model_1024'],
                512, 1024, 2048,
                merged_coords, shape_slat_sampler_params,
                {},
                denoise_strength,
                max_num_tokens,
            )
            hr_grid = new_res // 16
            mid_grid = 1024 // 16
            tex_rope_scale = hr_grid / mid_grid
            tex_saved_freqs = None
            if tex_rope_scale > 1.0:
                tex_saved_freqs = self._apply_ntk_rope_scaling(
                    self.models['tex_slat_flow_model_1024'], tex_rope_scale)
            try:
                tex_slat_new = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat_new, tex_slat_sampler_params
                )
            finally:
                if tex_saved_freqs is not None:
                    self._restore_rope_freqs(
                        self.models['tex_slat_flow_model_1024'], tex_saved_freqs)
        elif pipeline_type in ('1024_1536', '1024_2048'):
            target_res = {'1024_1536': 1536, '1024_2048': 2048}[pipeline_type]
            shape_slat_new, new_res = self.sample_shape_slat_cascade(
                cond_1024, cond_1024,
                self.models['shape_slat_flow_model_1024'],
                self.models['shape_slat_flow_model_1024'],
                1024, target_res,
                merged_coords, shape_slat_sampler_params,
                max_num_tokens=max_num_tokens,
                hr_rope_scale=target_res / 1024,
                denoise_strength=denoise_strength,
            )
            tex_slat_new = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat_new, tex_slat_sampler_params
            )
        else:
            raise ValueError(f"Unsupported pipeline type for recreate: {pipeline_type}")

        # ── Step 5: Stitch with original features ─────────────────
        print("[Recreate] Step 5/6: Stitching with original features...")
        hr_grid = new_res // 16

        # Create HR voxel mask on new coords
        hr_mask_new = self.create_voxel_mask_from_mesh(
            mask_vertices, shape_slat_new.coords, new_res, radius=mask_radius)

        # Also create HR mask on original coords
        hr_mask_orig = self.create_voxel_mask_from_mesh(
            mask_vertices, shape_slat.coords, new_res, radius=mask_radius)

        # Vectorized coord matching: new ↔ original
        new_xyz = shape_slat_new.coords[:, 1:]  # [N_new, 3]
        orig_xyz = shape_slat.coords[:, 1:].to(self.device)  # [N_orig, 3]
        orig_shape_feats = shape_slat.feats.to(self.device)
        orig_tex_feats = tex_slat.feats.to(self.device)

        def coords_to_hash(xyz, grid):
            return xyz[:, 0].long() * grid * grid + xyz[:, 1].long() * grid + xyz[:, 2].long()

        new_hash = coords_to_hash(new_xyz, hr_grid)
        orig_hash = coords_to_hash(orig_xyz, hr_grid)

        orig_hash_sorted, orig_sort_idx = orig_hash.sort()
        match_pos = torch.searchsorted(orig_hash_sorted, new_hash)
        match_pos = match_pos.clamp(max=orig_hash_sorted.shape[0] - 1)
        matched = orig_hash_sorted[match_pos] == new_hash
        orig_matched_idx = orig_sort_idx[match_pos]

        # Compute feather blend weights (mask distance based)
        mask_coord_indices = hr_mask_new.nonzero(as_tuple=True)[0]
        new_xyz_f = new_xyz.float()

        if mask_coord_indices.numel() > 0:
            mask_xyz_f = new_xyz_f[mask_coord_indices]
            chunk_size = 4096
            min_dists = torch.full((new_xyz_f.shape[0],), float('inf'), device=self.device)
            for i in range(0, mask_xyz_f.shape[0], chunk_size):
                mc = mask_xyz_f[i:i + chunk_size]
                diffs = (new_xyz_f.unsqueeze(1) - mc.unsqueeze(0)).abs()
                linf = diffs.max(dim=2).values.min(dim=1).values
                min_dists = torch.min(min_dists, linf)
            # 1.0 at mask, fading to 0.0 at feather distance
            blend_w = (1.0 - (min_dists / max(feather, 1)).clamp(0, 1)).unsqueeze(1)
        else:
            blend_w = torch.zeros(new_xyz_f.shape[0], 1, device=self.device)

        # Blend: new features where mask, original features where not mask
        final_shape_feats = shape_slat_new.feats.clone()
        final_tex_feats = tex_slat_new.feats.clone()

        # For non-mask coords with original match: blend
        has_orig = matched & ~hr_mask_new
        if has_orig.any():
            oi = orig_matched_idx[has_orig]
            w = blend_w[has_orig]
            final_shape_feats[has_orig] = (1 - w) * orig_shape_feats[oi] + w * final_shape_feats[has_orig]
            final_tex_feats[has_orig] = (1 - w) * orig_tex_feats[oi] + w * final_tex_feats[has_orig]

        # Also feather the boundary zone (matched + in feather range)
        boundary = matched & hr_mask_new & (blend_w.squeeze(1) < 1.0)
        if boundary.any():
            oi = orig_matched_idx[boundary]
            w = blend_w[boundary]
            final_shape_feats[boundary] = (1 - w) * orig_shape_feats[oi] + w * final_shape_feats[boundary]
            final_tex_feats[boundary] = (1 - w) * orig_tex_feats[oi] + w * final_tex_feats[boundary]

        # Add back original non-mask coords missing from new set
        new_hash_sorted = new_hash.sort()[0]
        orig_in_new_pos = torch.searchsorted(new_hash_sorted, orig_hash)
        orig_in_new_pos = orig_in_new_pos.clamp(max=new_hash_sorted.shape[0] - 1)
        orig_in_new = new_hash_sorted[orig_in_new_pos] == orig_hash
        missing = ~orig_in_new & ~hr_mask_orig
        n_missing = missing.sum().item()
        if n_missing > 0:
            print(f"  Restoring {n_missing} missing original non-mask voxels")
            missing_coords = shape_slat.coords[missing].to(self.device)
            missing_shape = orig_shape_feats[missing]
            missing_tex = orig_tex_feats[missing]
            all_coords = torch.cat([shape_slat_new.coords, missing_coords])
            final_shape_feats = torch.cat([final_shape_feats, missing_shape])
            final_tex_feats = torch.cat([final_tex_feats, missing_tex])
        else:
            all_coords = shape_slat_new.coords

        n_full_new = (hr_mask_new & (blend_w.squeeze(1) >= 1.0)).sum().item()
        n_blended = ((blend_w.squeeze(1) > 0) & (blend_w.squeeze(1) < 1.0)).sum().item()
        n_orig_kept = has_orig.sum().item() - n_blended
        print(f"  Voxels: {n_orig_kept} original, {n_blended} blended, "
              f"{n_full_new} new, {n_missing} restored")

        stitched_shape = SparseTensor(feats=final_shape_feats, coords=all_coords)
        stitched_tex = SparseTensor(feats=final_tex_feats, coords=all_coords)

        # ── Step 6: Decode ────────────────────────────────────────
        print("[Recreate] Step 6/6: Decoding...")
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(stitched_shape, stitched_tex, new_res)
        return out_mesh

    @torch.no_grad()
    def run_local_refine(
        self,
        image: Image.Image,
        mask_vertices: np.ndarray,
        seed: int = 42,
        refine_seed: int = 456,
        pipeline_type: Optional[str] = None,
        local_resolution: int = 1536,
        padding: int = 3,
        refine_denoise_strength: float = 0.5,
        refine_guidance_strength: float = 1.0,
        mask_radius: int = 1,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        denoise_strength: float = 1.0,
        max_num_tokens: int = 131072,
    ) -> Tuple[List[MeshWithVoxel], List[MeshWithVoxel]]:
        """
        Local grid refinement: re-generate the masked region at much higher
        voxel density by creating a local grid space covering only the mask AABB.

        This is a "local cascade" — the same pattern as the existing cascade
        (coarse→fine with decoder.upsample), but applied only to the masked region.

        Args:
            image: Input image prompt.
            mask_vertices: [M, 3] mask mesh vertices in GLB Y-up space.
            seed: Seed for reproducing the original generation.
            refine_seed: Seed for local refinement noise.
            pipeline_type: Pipeline type for original generation.
            local_resolution: Resolution for the local grid (e.g. 1536 = 96³).
            padding: AABB padding in coarse voxel units.
            refine_denoise_strength: Denoise strength for local refinement
                (0=keep coarse, 1=full denoise from noise).
            refine_guidance_strength: CFG strength for refinement (low = more freedom).
            mask_radius: Dilation radius for voxel mask.
            sparse_structure_sampler_params: For original generation.
            shape_slat_sampler_params: For original generation.
            tex_slat_sampler_params: For original generation.
            denoise_strength: For original cascade generation.
            max_num_tokens: Maximum tokens for original generation.

        Returns:
            Tuple of (original_meshes, local_refined_meshes).
            The local mesh is in global coordinates, covering only the AABB region.
        """
        pipeline_type = pipeline_type or self.default_pipeline_type
        local_grid = local_resolution // 16

        # Preprocess image once
        preprocessed = self.preprocess_image(image)

        # ── Step 1: Generate original ─────────────────────────────
        print("[LocalRefine] Step 1/6: Generating original model...")
        results = self.run(
            preprocessed, seed=seed, return_latent=True,
            preprocess_image=False,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            shape_slat_sampler_params=shape_slat_sampler_params,
            tex_slat_sampler_params=tex_slat_sampler_params,
            denoise_strength=denoise_strength,
            max_num_tokens=max_num_tokens,
        )
        orig_meshes, (shape_slat, tex_slat, res, _z_s) = results
        coarse_grid = res // 16
        print(f"  Original: res={res}, grid={coarse_grid}³, voxels={shape_slat.coords.shape[0]}")

        # ── Step 2: Extract mask AABB ─────────────────────────────
        print("[LocalRefine] Step 2/6: Extracting mask AABB...")
        mask = self.create_voxel_mask_from_mesh(
            mask_vertices, shape_slat.coords, res, radius=mask_radius
        )
        n_masked = mask.sum().item()
        print(f"  Mask: {n_masked}/{mask.shape[0]} voxels ({100*n_masked/mask.shape[0]:.1f}%)")

        masked_coords_xyz = shape_slat.coords[mask][:, 1:]
        aabb_min = masked_coords_xyz.min(dim=0).values - padding
        aabb_max = masked_coords_xyz.max(dim=0).values + padding
        aabb_min = aabb_min.clamp(min=0)
        aabb_max = aabb_max.clamp(max=coarse_grid - 1)
        aabb_size = (aabb_max - aabb_min + 1).float()
        print(f"  AABB: min={aabb_min.tolist()}, max={aabb_max.tolist()}, size={aabb_size.tolist()}")

        # ── Step 3: Extract AABB region + decoder.upsample ────────
        print("[LocalRefine] Step 3/6: Creating local sparse structure...")

        # Extract coarse voxels within AABB
        coords_xyz = shape_slat.coords[:, 1:]
        in_aabb = (
            (coords_xyz[:, 0] >= aabb_min[0]) & (coords_xyz[:, 0] <= aabb_max[0]) &
            (coords_xyz[:, 1] >= aabb_min[1]) & (coords_xyz[:, 1] <= aabb_max[1]) &
            (coords_xyz[:, 2] >= aabb_min[2]) & (coords_xyz[:, 2] <= aabb_max[2])
        )
        local_coarse_feats = shape_slat.feats[in_aabb]
        local_coarse_coords_xyz = coords_xyz[in_aabb]

        # Keep ORIGINAL coords for decoder.upsample() — decoder needs
        # proper spatial density to produce meaningful subdivisions.
        local_coarse_coords_orig = torch.cat([
            torch.zeros(local_coarse_coords_xyz.shape[0], 1, dtype=torch.int32, device=self.device),
            local_coarse_coords_xyz,
        ], dim=1)

        local_coarse_slat = SparseTensor(
            feats=local_coarse_feats,
            coords=local_coarse_coords_orig,
        )
        print(f"  Coarse voxels in AABB: {local_coarse_coords_orig.shape[0]}")

        # decoder.upsample() with original coords (proper spatial structure)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        raw_hr_coords = self.models['shape_slat_decoder'].upsample(local_coarse_slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False

        # Quantize raw HR coords to LOCAL grid [0, local_grid-1].
        # Raw coords are at coarse_grid*16 = res scale (e.g. 1536).
        # Map the AABB portion of that scale to [0, local_grid).
        aabb_min_fine = aabb_min.float() * 16  # AABB min in fine (res) scale
        aabb_size_fine = aabb_size * 16         # AABB size in fine (res) scale
        quant_coords = torch.cat([
            raw_hr_coords[:, :1],
            ((raw_hr_coords[:, 1:].float() - aabb_min_fine.unsqueeze(0) + 0.5)
             / aabb_size_fine.unsqueeze(0) * local_grid).int().clamp(0, local_grid - 1),
        ], dim=1)
        local_hr_coords = quant_coords.unique(dim=0)

        # Also remap coarse coords to local grid for feature inheritance
        local_coarse_in_local = torch.cat([
            torch.zeros(local_coarse_coords_xyz.shape[0], 1, dtype=torch.int32, device=self.device),
            ((local_coarse_coords_xyz.float() - aabb_min.unsqueeze(0) + 0.5)
             / aabb_size.unsqueeze(0) * local_grid).int().clamp(0, local_grid - 1),
        ], dim=1)
        local_coarse_slat_remapped = SparseTensor(
            feats=local_coarse_feats,
            coords=local_coarse_in_local,
        )

        print(f"  Local HR voxels: {local_hr_coords.shape[0]} (target grid: {local_grid}³)")

        # ── Step 4: Seed coarse features + partial denoise ────────
        print("[LocalRefine] Step 4/6: Shape refinement...")

        shape_std = torch.tensor(self.shape_slat_normalization['std'])[None].to(self.device)
        shape_mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(self.device)

        shape_flow_model = (
            self.models['shape_slat_flow_model_512'] if pipeline_type == '512'
            else self.models['shape_slat_flow_model_1024']
        )

        # Feature inheritance: coarse (remapped to local grid) → local HR (nearest-neighbor)
        upsampled, found_mask = self._upsample_slat_features(
            local_coarse_slat_remapped, local_hr_coords, local_grid, local_grid
        )

        # Normalize and prepare noisy input (cascade partial denoise pattern)
        x0 = (upsampled.feats - shape_mean) / shape_std
        C_model = shape_flow_model.in_channels
        C_feat = x0.shape[1]
        if C_feat < C_model:
            x0 = torch.cat([x0, torch.zeros(x0.shape[0], C_model - C_feat, device=self.device)], dim=1)
        elif C_feat > C_model:
            x0 = x0[:, :C_model]

        torch.manual_seed(refine_seed)
        t = refine_denoise_strength
        eps = torch.randn_like(x0)
        if t >= 1.0:
            noisy_input = eps
            start_t = 1.0
        else:
            noisy_input = (1 - t) * x0 + t * eps
            noisy_input[~found_mask] = eps[~found_mask]
            start_t = t

        print(f"  denoise_strength={t:.2f}, start_t={start_t:.2f}, tokens={local_hr_coords.shape[0]}")

        noise = SparseTensor(
            feats=noisy_input,
            coords=local_hr_coords,
        )

        # Get conditioning
        cond_res = 512 if pipeline_type == '512' else 1024
        cond = self.get_cond([preprocessed], cond_res)

        # NTK RoPE scaling if local grid exceeds model native
        model_native_grid = 32 if pipeline_type == '512' else 64
        saved_freqs = None
        if local_grid > model_native_grid:
            rope_scale = local_grid / model_native_grid
            saved_freqs = self._apply_ntk_rope_scaling(shape_flow_model, rope_scale)
            print(f"  NTK RoPE scaling: {model_native_grid}³→{local_grid}³ (scale={rope_scale:.1f})")

        try:
            refine_sampler_params = {
                **self.shape_slat_sampler_params,
                **shape_slat_sampler_params,
                'guidance_strength': refine_guidance_strength,
            }
            if self.low_vram:
                shape_flow_model.to(self.device)

            local_shape_slat = self.shape_slat_sampler.sample(
                shape_flow_model, noise,
                **cond, **refine_sampler_params,
                start_t=start_t,
                verbose=True,
                tqdm_desc="Local refine shape SLat",
            ).samples

            if self.low_vram:
                shape_flow_model.cpu()

            local_shape_slat = local_shape_slat * shape_std + shape_mean
        finally:
            if saved_freqs is not None:
                self._restore_rope_freqs(shape_flow_model, saved_freqs)

        # ── Step 5: Texture refinement ────────────────────────────
        print("[LocalRefine] Step 5/6: Texture refinement...")

        tex_std = torch.tensor(self.tex_slat_normalization['std'])[None].to(self.device)
        tex_mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(self.device)

        # Normalize refined shape as concat_cond
        local_shape_norm = local_shape_slat.replace(
            feats=(local_shape_slat.feats - shape_mean) / shape_std
        )

        tex_flow_model = (
            self.models['tex_slat_flow_model_512'] if pipeline_type == '512'
            else self.models['tex_slat_flow_model_1024']
        )

        tex_saved_freqs = None
        if local_grid > model_native_grid:
            tex_rope_scale = local_grid / model_native_grid
            tex_saved_freqs = self._apply_ntk_rope_scaling(tex_flow_model, tex_rope_scale)

        try:
            torch.manual_seed(refine_seed + 1)
            tex_in_channels = (
                tex_flow_model.in_channels if isinstance(tex_flow_model, nn.Module)
                else tex_flow_model[0].in_channels
            )
            tex_noise_channels = tex_in_channels - local_shape_norm.feats.shape[1]
            tex_noise_feats = torch.randn(
                local_shape_norm.coords.shape[0], tex_noise_channels
            ).to(self.device)
            tex_noise = local_shape_norm.replace(feats=tex_noise_feats)

            tex_sampler_params = {
                **self.tex_slat_sampler_params,
                **tex_slat_sampler_params,
            }
            if self.low_vram:
                tex_flow_model.to(self.device)

            local_tex_slat = self.tex_slat_sampler.sample(
                tex_flow_model, tex_noise,
                concat_cond=local_shape_norm,
                **cond, **tex_sampler_params,
                verbose=True,
                tqdm_desc="Local refine texture SLat",
            ).samples

            if self.low_vram:
                tex_flow_model.cpu()

            local_tex_slat = local_tex_slat * tex_std + tex_mean
        finally:
            if tex_saved_freqs is not None:
                self._restore_rope_freqs(tex_flow_model, tex_saved_freqs)

        # ── Step 6: Decode + coordinate transform ─────────────────
        print("[LocalRefine] Step 6/6: Decoding local mesh...")
        torch.cuda.empty_cache()
        local_meshes = self.decode_latent(local_shape_slat, local_tex_slat, local_resolution)

        # Compute AABB world-space info (for future stitching)
        aabb_world_min = -0.5 + aabb_min.float() / coarse_grid
        aabb_world_size = aabb_size / coarse_grid

        print(f"  Local mesh vertices: {local_meshes[0].vertices.shape[0]:,}")
        print(f"  AABB world: min={aabb_world_min.tolist()}, size={aabb_world_size.tolist()}")
        print(f"  (Local mesh kept in [-0.5,0.5]³ space; transform info saved for stitching)")

        return orig_meshes, local_meshes
