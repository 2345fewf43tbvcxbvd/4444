import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from .analogizer import Analogizer
from .fft_surgeon import FFTSurgeon
from .vlm_caption import HearmemanAI_Prompter


class Realism_AutoWB:
    """
    Auto White Balance node that corrects unnatural color casts in AI-generated images.
    Uses Gray World algorithm with manual temperature and tint adjustments.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature_shift": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "tint_shift": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "mix_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_autowb"
    CATEGORY = "image/postprocessing"
    
    @torch.inference_mode()
    def apply_autowb(self, image, temperature_shift, tint_shift, mix_strength):
        # Clone input to prevent modifying cached tensor
        image = image.clone()
        # Convert from [B, H, W, 3] to [B, 3, H, W] for easier channel operations
        img = image.permute(0, 3, 1, 2)
        
        # Calculate per-channel means
        r_mean = img[:, 0, :, :].mean()
        g_mean = img[:, 1, :, :].mean()
        b_mean = img[:, 2, :, :].mean()
        
        # Gray World target (average of all channel means)
        gray_target = (r_mean + g_mean + b_mean) / 3.0
        
        # Avoid division by zero
        epsilon = 1e-8
        r_scale = gray_target / (r_mean + epsilon)
        b_scale = gray_target / (b_mean + epsilon)
        
        # Create corrected image
        corrected = img.clone()
        
        # Scale Red and Blue channels to match Gray World
        corrected[:, 0, :, :] *= r_scale
        corrected[:, 2, :, :] *= b_scale
        
        # Apply manual temperature shift (affects R/B bias)
        # Positive temperature = warmer (more red, less blue)
        temp_factor = 1.0 + temperature_shift
        corrected[:, 0, :, :] *= temp_factor
        corrected[:, 2, :, :] *= (2.0 - temp_factor)
        
        # Apply manual tint shift (affects G bias)
        # Positive tint = more green, negative = more magenta
        tint_factor = 1.0 + tint_shift
        corrected[:, 1, :, :] *= tint_factor
        
        # Clamp to valid range
        corrected = torch.clamp(corrected, 0.0, 1.0)
        
        # Mix with original
        result = mix_strength * corrected + (1.0 - mix_strength) * img
        
        # Convert back to [B, H, W, 3]
        result = result.permute(0, 2, 3, 1)
        
        return (result,)


class Realism_MicroContrast:
    """
    Micro Contrast node that enhances local texture detail without affecting global contrast.
    Uses unsharp mask technique with variance-based protection for smooth areas.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detail_scale": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "contrast_amount": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "protection_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_microcontrast"
    CATEGORY = "image/postprocessing"
    
    @torch.inference_mode()
    def apply_microcontrast(self, image, detail_scale, contrast_amount, protection_threshold):
        # Clone input to prevent modifying cached tensor
        image = image.clone()
        # Convert from [B, H, W, 3] to [B, 3, H, W]
        img = image.permute(0, 3, 1, 2)
        
        # Calculate kernel size from detail_scale (must be odd)
        kernel_size = max(3, detail_scale * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Calculate sigma for Gaussian blur (roughly 1/3 of kernel size)
        sigma = detail_scale / 3.0
        
        # Apply Gaussian blur
        blurred = gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        
        # Calculate detail map (high-frequency components)
        detail = img - blurred
        
        # Calculate local variance for protection mask
        # Compute variance across channels for each pixel
        img_mean = img.mean(dim=1, keepdim=True)
        variance = ((img - img_mean) ** 2).mean(dim=1, keepdim=True)
        
        # Create protection mask (1.0 where variance > threshold, 0.0 otherwise)
        protection_mask = (variance > protection_threshold).float()
        
        # Apply contrast boost with protection
        enhanced_detail = detail * contrast_amount * protection_mask
        
        # Add enhanced detail back to original
        result = img + enhanced_detail
        
        # Clamp to valid range
        result = torch.clamp(result, 0.0, 1.0)
        
        # Convert back to [B, H, W, 3]
        result = result.permute(0, 2, 3, 1)
        
        return (result,)


class Realism_AdaptiveGrain:
    """
    Adaptive Grain node that simulates film emulsion grain with luminance-based intensity mapping.
    Grain intensity varies based on shadows, midtones, and highlights.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grain_size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1
                }),
                "overall_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "shadow_intensity": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "midtone_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "highlight_intensity": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "monochrome_grain": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_adaptive_grain"
    CATEGORY = "image/postprocessing"
    
    @torch.inference_mode()
    def apply_adaptive_grain(self, image, grain_size, overall_strength, shadow_intensity, 
                            midtone_intensity, highlight_intensity, monochrome_grain):
        # Clone input to prevent modifying cached tensor
        image = image.clone()
        # Convert from [B, H, W, 3] to [B, 3, H, W]
        img = image.permute(0, 3, 1, 2)
        
        # Generate Gaussian noise
        noise = torch.randn_like(img) * grain_size
        
        # Apply monochrome if enabled
        if monochrome_grain:
            noise_gray = noise.mean(dim=1, keepdim=True)
            noise = noise_gray.expand_as(noise)
        
        # Calculate luminance map
        luminance = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
        luminance = luminance.unsqueeze(1)  # [B, 1, H, W]
        
        # Map luminance to intensity curve with smooth interpolation
        # Shadows: luminance < 0.33
        # Midtones: 0.33 <= luminance < 0.66
        # Highlights: luminance >= 0.66
        
        # Create smooth transitions between zones
        shadow_zone = torch.clamp((0.33 - luminance) / 0.33, 0.0, 1.0)
        midtone_zone = torch.clamp(1.0 - torch.abs(luminance - 0.5) / 0.17, 0.0, 1.0)
        highlight_zone = torch.clamp((luminance - 0.66) / 0.34, 0.0, 1.0)
        
        # Normalize zones so they sum to 1.0 for smooth blending
        zone_sum = shadow_zone + midtone_zone + highlight_zone + 1e-8
        shadow_weight = shadow_zone / zone_sum
        midtone_weight = midtone_zone / zone_sum
        highlight_weight = highlight_zone / zone_sum
        
        # Calculate intensity map
        intensity_map = (shadow_weight * shadow_intensity + 
                        midtone_weight * midtone_intensity + 
                        highlight_weight * highlight_intensity)
        
        # Scale noise by intensity map and overall strength
        # Normalize noise to reasonable range for overlay blend
        noise_normalized = torch.tanh(noise) * 0.5  # [-0.5, 0.5]
        scaled_noise = noise_normalized * intensity_map * overall_strength
        
        # Convert noise to [0, 1] range for overlay blend calculation
        # Center the noise around 0.5 (neutral for overlay)
        noise_blend = 0.5 + scaled_noise
        noise_blend = torch.clamp(noise_blend, 0.0, 1.0)
        
        # Apply Overlay blend mode
        # Overlay: if base < 0.5: 2 * base * blend, else: 1 - 2 * (1 - base) * (1 - blend)
        mask_low = (img < 0.5).float()
        mask_high = 1.0 - mask_low
        
        # Low range overlay: 2 * img * noise_blend
        result_low = 2.0 * img * noise_blend
        
        # High range overlay: 1 - 2 * (1 - img) * (1 - noise_blend)
        result_high = 1.0 - 2.0 * (1.0 - img) * (1.0 - noise_blend)
        
        # Combine
        result = mask_low * result_low + mask_high * result_high
        
        # Clamp to valid range
        result = torch.clamp(result, 0.0, 1.0)
        
        # Convert back to [B, H, W, 3]
        result = result.permute(0, 2, 3, 1)
        
        return (result,)


class Realism_LensEffects:
    """
    Lens Effects node that simulates physical optical flaws: chromatic aberration, halation, and blur vignette.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "chromatic_aberration": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1
                }),
                "blur_vignette": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "halation_threshold": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "halation_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lens_effects"
    CATEGORY = "image/postprocessing"
    
    @torch.inference_mode()
    def apply_lens_effects(self, image, chromatic_aberration, blur_vignette, 
                          halation_threshold, halation_intensity):
        # Clone input to prevent modifying cached tensor
        image = image.clone()
        # Convert from [B, H, W, 3] to [B, 3, H, W]
        img = image.permute(0, 3, 1, 2)
        B, C, H, W = img.shape
        
        # Create coordinate grids
        y_coords = torch.arange(H, dtype=torch.float32, device=img.device)
        x_coords = torch.arange(W, dtype=torch.float32, device=img.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Center coordinates
        center_y = H / 2.0
        center_x = W / 2.0
        
        # Distance from center (normalized)
        y_dist = (y_grid - center_y) / max(H, W)
        x_dist = (x_grid - center_x) / max(H, W)
        dist = torch.sqrt(x_dist ** 2 + y_dist ** 2)
        max_dist = torch.sqrt(torch.tensor(0.5, device=img.device))  # Max normalized distance
        normalized_dist = dist / max_dist
        
        result = img.clone()
        
        # Apply Chromatic Aberration
        if chromatic_aberration > 0:
            # Create shift grids for Red (outward) and Blue (inward)
            # Shift amount in pixels, proportional to distance from center
            max_dim = max(H, W)
            shift_scale = chromatic_aberration / max_dim
            
            # Red shifts outward: positive shift
            red_shift_y = y_dist * shift_scale * max_dim
            red_shift_x = x_dist * shift_scale * max_dim
            
            # Blue shifts inward: negative shift
            blue_shift_y = -y_dist * shift_scale * max_dim
            blue_shift_x = -x_dist * shift_scale * max_dim
            
            # Create grid for grid_sample (normalized coordinates [-1, 1])
            # y_grid and x_grid are [H, W], shifts are [H, W]
            red_grid_y = (y_grid + red_shift_y) / (H - 1) * 2.0 - 1.0
            red_grid_x = (x_grid + red_shift_x) / (W - 1) * 2.0 - 1.0
            blue_grid_y = (y_grid + blue_shift_y) / (H - 1) * 2.0 - 1.0
            blue_grid_x = (x_grid + blue_shift_x) / (W - 1) * 2.0 - 1.0
            
            # Stack grids for grid_sample: [H, W, 2] where last dim is [x, y]
            # Then expand to [B, H, W, 2]
            red_grid = torch.stack([red_grid_x, red_grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            blue_grid = torch.stack([blue_grid_x, blue_grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            
            # Sample Red channel (outward)
            red_channel = img[:, 0:1, :, :]
            red_shifted = F.grid_sample(red_channel, red_grid, mode='bilinear', padding_mode='border', align_corners=True)
            
            # Sample Blue channel (inward)
            blue_channel = img[:, 2:3, :, :]
            blue_shifted = F.grid_sample(blue_channel, blue_grid, mode='bilinear', padding_mode='border', align_corners=True)
            
            # Replace channels
            result[:, 0, :, :] = red_shifted.squeeze(1)
            result[:, 2, :, :] = blue_shifted.squeeze(1)
        
        # Apply Halation
        if halation_intensity > 0:
            # Isolate highlights
            highlights = (result > halation_threshold).float()
            
            # Blur highlights heavily
            blurred_highlights = gaussian_blur(highlights, kernel_size=[51, 51], sigma=[15.0, 15.0])
            
            # Tint red/orange: [R=1.0, G=0.5, B=0.2]
            tinted = blurred_highlights.clone()
            tinted[:, 0, :, :] = blurred_highlights[:, 0, :, :] * 1.0
            tinted[:, 1, :, :] = blurred_highlights[:, 1, :, :] * 0.5
            tinted[:, 2, :, :] = blurred_highlights[:, 2, :, :] * 0.2
            
            # Screen blend: 1 - (1 - image) * (1 - tinted * intensity)
            one_minus_img = 1.0 - result
            one_minus_tinted = 1.0 - tinted * halation_intensity
            halation_result = 1.0 - one_minus_img * one_minus_tinted
            
            result = halation_result
        
        # Apply Blur Vignette
        if blur_vignette > 0:
            # Create radial gradient mask (1.0 at center, 0.0 at edges)
            vignette_mask = 1.0 - torch.clamp(normalized_dist, 0.0, 1.0)
            vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Blur entire image
            blurred_img = gaussian_blur(result, kernel_size=[31, 31], sigma=[10.0, 10.0])
            
            # Mix original and blurred based on vignette mask
            result = vignette_mask * result + (1.0 - vignette_mask) * (result * (1.0 - blur_vignette) + blurred_img * blur_vignette)
        
        # Clamp to valid range
        result = torch.clamp(result, 0.0, 1.0)
        
        # Convert back to [B, H, W, 3]
        result = result.permute(0, 2, 3, 1)
        
        return (result,)


class Realism_SpectrumMatch:
    """
    Spectrum Match node that transfers the global frequency profile from a reference photo to the target image.
    Uses per-channel FFT processing with amplitude ratio clamping to prevent artifacts.
    
    FIXED: Now processes each RGB channel separately and uses ratio-based blending
    with clamping to prevent the horizontal banding artifacts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Overall effect strength. Start low (0.3-0.5)"
                }),
                "high_pass_cutoff": ("INT", {
                    "default": 80,
                    "min": 10,
                    "max": 500,
                    "step": 5,
                    "tooltip": "Frequency cutoff in pixels. Lower = affect more frequencies"
                }),
                "amplitude_clamp": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Max amplitude ratio. Lower = safer but weaker effect"
                }),
                "preserve_color": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process luminance only, preserve original colors"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_spectrum_match"
    CATEGORY = "image/postprocessing"
    
    @torch.inference_mode()
    def apply_spectrum_match(self, target_image, reference_image, strength, high_pass_cutoff, amplitude_clamp, preserve_color):
        # Clone inputs to prevent modifying cached tensors
        target = target_image.clone().permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        reference = reference_image.clone().permute(0, 3, 1, 2)
        
        B, C, H, W = target.shape
        _, _, H_ref, W_ref = reference.shape
        
        # Resize reference to match target dimensions
        if H != H_ref or W != W_ref:
            # Aspect-ratio preserving resize with center crop
            scale_h = H / H_ref
            scale_w = W / W_ref
            scale = max(scale_h, scale_w)
            
            new_H = int(H_ref * scale)
            new_W = int(W_ref * scale)
            reference = F.interpolate(reference, size=(new_H, new_W), mode='bicubic', align_corners=False)
            
            crop_y = (new_H - H) // 2
            crop_x = (new_W - W) // 2
            reference = reference[:, :, crop_y:crop_y+H, crop_x:crop_x+W]
        
        # Create frequency-domain mask (Gaussian high-pass)
        Y, X = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=target.device),
            torch.arange(W, dtype=torch.float32, device=target.device),
            indexing='ij'
        )
        center_y, center_x = H // 2, W // 2
        dist = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # High-pass mask: 0 at center (low freq), 1 at edges (high freq)
        sigma = float(high_pass_cutoff)
        hp_mask = 1.0 - torch.exp(-(dist**2) / (2 * sigma**2 + 1e-8))
        hp_mask = hp_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        if preserve_color:
            # Process luminance channel only
            # Convert to luminance
            target_lum = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
            ref_lum = 0.299 * reference[:, 0:1] + 0.587 * reference[:, 1:2] + 0.114 * reference[:, 2:3]
            
            # Process single luminance channel
            result_lum = self._process_channel(target_lum, ref_lum, hp_mask, strength, amplitude_clamp)
            
            # Apply luminance change to RGB while preserving color ratios
            # Avoid division by zero
            original_lum = target_lum.clamp(min=1e-6)
            lum_ratio = result_lum / original_lum
            
            # Apply ratio to all channels (preserves color relationships)
            result = target * lum_ratio
            result = torch.clamp(result, 0.0, 1.0)
        else:
            # Process each RGB channel separately
            result_channels = []
            for c in range(C):
                target_ch = target[:, c:c+1, :, :]
                ref_ch = reference[:, c:c+1, :, :]
                result_ch = self._process_channel(target_ch, ref_ch, hp_mask, strength, amplitude_clamp)
                result_channels.append(result_ch)
            
            result = torch.cat(result_channels, dim=1)
            result = torch.clamp(result, 0.0, 1.0)
        
        # Convert back to [B, H, W, C]
        return (result.permute(0, 2, 3, 1),)
    
    def _process_channel(self, target_ch, ref_ch, hp_mask, strength, amplitude_clamp):
        """
        Process a single channel through FFT spectrum matching.
        Uses ratio-based amplitude transfer with clamping to prevent artifacts.
        """
        # FFT
        target_fft = torch.fft.fft2(target_ch)
        ref_fft = torch.fft.fft2(ref_ch)
        
        # Shift to center DC
        target_fft_shifted = torch.fft.fftshift(target_fft)
        ref_fft_shifted = torch.fft.fftshift(ref_fft)
        
        # Extract amplitude and phase
        target_amp = torch.abs(target_fft_shifted)
        target_phase = torch.angle(target_fft_shifted)
        ref_amp = torch.abs(ref_fft_shifted)
        
        # Calculate amplitude ratio (reference / target)
        # Clamp to prevent extreme values that cause artifacts
        epsilon = 1e-8
        amp_ratio = ref_amp / (target_amp + epsilon)
        amp_ratio = torch.clamp(amp_ratio, 1.0 / amplitude_clamp, amplitude_clamp)
        
        # Apply ratio only in high-frequency regions (masked)
        # In low-freq regions (mask=0), keep ratio=1 (no change)
        # In high-freq regions (mask=1), apply the clamped ratio
        effective_ratio = 1.0 + (amp_ratio - 1.0) * hp_mask * strength
        
        # Apply ratio to target amplitude
        blended_amp = target_amp * effective_ratio
        
        # Reconstruct complex FFT
        new_fft_shifted = blended_amp * torch.exp(1j * target_phase)
        
        # Inverse shift and FFT
        new_fft = torch.fft.ifftshift(new_fft_shifted)
        result = torch.fft.ifft2(new_fft).real
        
        # Soft normalization: match mean and std of original
        # This prevents brightness shifts while allowing texture transfer
        target_mean = target_ch.mean()
        target_std = target_ch.std()
        result_mean = result.mean()
        result_std = result.std() + epsilon
        
        # Normalize to match original statistics
        result = (result - result_mean) / result_std * target_std + target_mean
        
        return result


# Node registration
NODE_CLASS_MAPPINGS = {
    "Realism_AutoWB": Realism_AutoWB,
    "Realism_MicroContrast": Realism_MicroContrast,
    "Realism_AdaptiveGrain": Realism_AdaptiveGrain,
    "Realism_LensEffects": Realism_LensEffects,
    "Realism_SpectrumMatch": Realism_SpectrumMatch,
    "Analogizer": Analogizer,
    "FFTSurgeon": FFTSurgeon,
    "HearmemanAI_Prompter": HearmemanAI_Prompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Realism_AutoWB": "Realism AutoWB",
    "Realism_MicroContrast": "Realism MicroContrast",
    "Realism_AdaptiveGrain": "Realism AdaptiveGrain",
    "Realism_LensEffects": "Realism LensEffects",
    "Realism_SpectrumMatch": "Realism SpectrumMatch",
    "Analogizer": "Analog Pipeline (Pro)",
    "FFTSurgeon": "FFT Surgeon (Precision)",
    "HearmemanAI_Prompter": "HearmemanAI Prompter",
}

# Web directory for custom UI styling
WEB_DIRECTORY = "web"

# Export all required symbols
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

