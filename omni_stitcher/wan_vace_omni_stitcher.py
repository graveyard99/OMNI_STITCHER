import gc
import math
from typing import Dict, List, Optional, Tuple

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers


def slice_tensor_safe(tensor: Optional[torch.Tensor], start_idx: int, length: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None

    total_len = tensor.shape[0]
    end_idx = start_idx + length

    if end_idx <= total_len:
        return tensor[start_idx:end_idx]

    indices = [(i % total_len) for i in range(start_idx, end_idx)]
    return tensor[indices]


def apply_pseudo_mask(image: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return image

    if mask.dim() == 3:
        mask = mask.unsqueeze(-1)

    white = torch.ones_like(image)
    return torch.where(mask > 0.5, white, image)


def cross_fade(tensor_a: torch.Tensor, tensor_b: torch.Tensor, frames: int) -> torch.Tensor:
    if frames <= 0:
        return tensor_b

    alphas = torch.linspace(0, 1, frames, device=tensor_a.device)
    alphas = 1 / (1 + torch.exp(-10 * (alphas - 0.5)))
    alphas = alphas.view(frames, 1, 1, 1)
    return tensor_a * (1 - alphas) + tensor_b * alphas


class WanVACE_OmniStitcher:
    """
    Unified orchestration node for Wan 2.1 VACE workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "reference_image": ("IMAGE",),
                "total_frames": ("INT", {"default": 152, "min": 16, "max": 10000}),
                "segment_length": ("INT", {"default": 81, "min": 16, "max": 200}),
                "context_overlap": ("INT", {"default": 10, "min": 0, "max": 40}),
                "video_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "reverse_order": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "control_video_1": ("IMAGE",),
                "control_mask_1": ("MASK",),
                "control_video_2": ("IMAGE",),
                "control_mask_2": ("MASK",),
                "control_video_3": ("IMAGE",),
                "control_mask_3": ("MASK",),
                "start_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("stitched_video", "raw_latents")
    FUNCTION = "generate_sequence"
    CATEGORY = "Wan2.1/VACE"

    def generate_sequence(
        self,
        model,
        vae,
        positive,
        negative,
        reference_image,
        total_frames,
        segment_length,
        context_overlap,
        video_strength,
        reverse_order,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        control_video_1=None,
        control_mask_1=None,
        control_video_2=None,
        control_mask_2=None,
        control_video_3=None,
        control_mask_3=None,
        start_image=None,
    ):
        stride = segment_length - context_overlap
        if stride <= 0:
            raise ValueError("segment_length must be larger than context_overlap")

        num_segments = max(1, math.ceil((total_frames - context_overlap) / stride))
        loop_sequence = range(num_segments - 1, -1, -1) if reverse_order else range(num_segments)

        segment_storage: Dict[int, torch.Tensor] = {}
        segment_latents: Dict[int, torch.Tensor] = {}

        for seg_idx in loop_sequence:
            global_start = seg_idx * stride
            current_seed = seed + seg_idx

            def prepare_control(vid, mask):
                if vid is None:
                    return None
                c_slice = slice_tensor_safe(vid, global_start, segment_length)
                c_mask_slice = slice_tensor_safe(mask, global_start, segment_length) if mask is not None else None
                return (c_slice, c_mask_slice)

            c1 = prepare_control(control_video_1, control_mask_1)
            c2 = prepare_control(control_video_2, control_mask_2)
            c3 = prepare_control(control_video_3, control_mask_3)

            current_start_cond = None
            current_end_cond = None

            if reverse_order:
                if seg_idx == num_segments - 1:
                    current_end_cond = reference_image
                else:
                    prev_seg_pixels = segment_storage.get(seg_idx + 1)
                    if prev_seg_pixels is not None:
                        current_end_cond = prev_seg_pixels[0:1]

                if seg_idx == 0:
                    current_start_cond = start_image
            else:
                if seg_idx == 0:
                    current_start_cond = start_image if start_image is not None else reference_image
                else:
                    prev_seg_pixels = segment_storage.get(seg_idx - 1)
                    if prev_seg_pixels is not None:
                        current_start_cond = prev_seg_pixels[-1:]

            seg_positive = [p.copy() for p in positive]
            for p in seg_positive:
                if current_start_cond is not None:
                    p["start_img"] = current_start_cond
                if current_end_cond is not None:
                    p["end_img"] = current_end_cond

                if c1:
                    p["control_1"] = c1
                if c2:
                    p["control_2"] = c2
                if c3:
                    p["control_3"] = c3

            comfy.model_management.soft_empty_cache()
            gc.collect()

            _, height, width, _ = reference_image.shape
            latent_image = {"samples": torch.zeros((1, 4, segment_length // 4, height // 8, width // 8))}

            segment_denoise = denoise * video_strength
            samples = comfy.sample.sample(
                model,
                noise=None,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=seg_positive,
                negative=negative,
                latent_image=latent_image,
                denoise=segment_denoise,
                seed=current_seed,
            )

            decoded_pixels = vae.decode(samples["samples"])
            segment_storage[seg_idx] = decoded_pixels.cpu()
            segment_latents[seg_idx] = samples["samples"].cpu()

            del samples
            del decoded_pixels

        final_video_frames: List[torch.Tensor] = []
        ordered_segments = range(num_segments)

        for i in ordered_segments:
            curr_pixels = segment_storage[i]

            if i == 0:
                if num_segments == 1:
                    final_video_frames.append(curr_pixels)
                else:
                    safe_zone = curr_pixels[:-context_overlap]
                    final_video_frames.append(safe_zone)
            else:
                prev_pixels = segment_storage[i - 1]
                overlap_prev = prev_pixels[-context_overlap:]
                overlap_curr = curr_pixels[:context_overlap]
                blended_overlap = cross_fade(overlap_prev, overlap_curr, context_overlap)
                final_video_frames.append(blended_overlap)

                if i < num_segments - 1:
                    safe_zone = curr_pixels[context_overlap:-context_overlap]
                else:
                    safe_zone = curr_pixels[context_overlap:]
                final_video_frames.append(safe_zone)

        full_video = torch.cat(final_video_frames, dim=0)[:total_frames]
        raw_latents = torch.cat([segment_latents[i] for i in ordered_segments], dim=2)

        return (full_video, {"samples": raw_latents})


NODE_CLASS_MAPPINGS = {"WanVACE_OmniStitcher": WanVACE_OmniStitcher}
NODE_DISPLAY_NAME_MAPPINGS = {"WanVACE_OmniStitcher": "Wan VACE OmniStitcher"}
