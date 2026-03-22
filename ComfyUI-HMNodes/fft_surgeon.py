import torch
import torch.nn.functional as F
import torch.fft


class FFTSurgeon:
    """
    FFT Surgeon - Spectrum-based naturalization for AI images.
    
    NEW APPROACH: Instead of trying to REMOVE AI artifacts (which causes blur),
    this node INJECTS natural high-frequency characteristics.
    
    Strategy:
    1. Analyze the image's frequency spectrum
    2. Identify unnaturally smooth/clean frequency bands
    3. Inject subtle, natural-looking high-frequency texture
    4. Match the frequency "slope" of real photographs
    
    Real photos have a characteristic 1/f frequency falloff (pink noise).
    AI images often deviate from this, especially in smooth areas.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # Spectrum Slope Correction
                "slope_correction": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Correct frequency slope toward natural 1/f distribution. 0=Off, 0.3-0.5=Subtle"
                }),
                # High-Frequency Texture Injection
                "texture_injection": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Inject natural high-freq texture. Key for flat areas! 0.1-0.2=Subtle"
                }),
                "texture_frequency": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.3,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Where to start texture injection. 0.6 = top 40% of frequencies"
                }),
                # Micro-Contrast Enhancement
                "micro_contrast": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Enhance micro-detail that AI tends to smooth over"
                }),
                "contrast_frequency": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.15,
                    "max": 0.6,
                    "step": 0.05,
                    "tooltip": "Center frequency for micro-contrast boost"
                }),
                # Processing Options
                "preserve_luminance": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process luminance only to prevent color shifts"
                }),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_spectrum_surgery"
    CATEGORY = "SecretLab"

    def apply_spectrum_surgery(self, image, slope_correction, texture_injection,
                                texture_frequency, micro_contrast, contrast_frequency,
                                preserve_luminance, reference_image=None):
        # Setup: [B,H,W,C] -> [B,C,H,W]
        x = image.permute(0, 3, 1, 2).clone()
        B, C, H, W = x.shape
        
        # Create frequency magnitude grid
        fy = torch.linspace(-0.5, 0.5, H, device=x.device).view(-1, 1)
        fx = torch.linspace(-0.5, 0.5, W, device=x.device).view(1, -1)
        freq_mag = torch.sqrt(fx**2 + fy**2)
        freq_mag = freq_mag.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        if preserve_luminance:
            # Process only luminance channel
            luminance = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            color_ratios = x / (luminance + 1e-6)
            
            # Process luminance
            lum_processed = self.process_channel(
                luminance, freq_mag, slope_correction, texture_injection,
                texture_frequency, micro_contrast, contrast_frequency,
                reference_image, H, W
            )
            
            # Reconstruct RGB
            result = lum_processed * color_ratios
        else:
            # Process each channel
            result = self.process_channel(
                x, freq_mag, slope_correction, texture_injection,
                texture_frequency, micro_contrast, contrast_frequency,
                reference_image, H, W
            )
        
        # Match original statistics to prevent brightness shifts
        result = self.match_statistics(result, x)
        
        # Clamp and return
        result = torch.clamp(result, 0.0, 1.0)
        return (result.permute(0, 2, 3, 1),)

    def process_channel(self, channel, freq_mag, slope_correction, texture_injection,
                        texture_frequency, micro_contrast, contrast_frequency,
                        reference_image, H, W):
        """Process image channel through frequency domain modifications."""
        
        # FFT
        fft = torch.fft.fft2(channel)
        fft_shifted = torch.fft.fftshift(fft)
        
        amplitude = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)
        
        # === 1. SLOPE CORRECTION ===
        # Natural images follow ~1/f amplitude falloff (pink noise)
        # AI images often have different slopes, especially too flat in high-freq
        if slope_correction > 0:
            amplitude = self.apply_slope_correction(amplitude, freq_mag, slope_correction)
        
        # === 2. MICRO-CONTRAST BOOST ===
        # Enhance mid-high frequencies where fine detail lives
        if micro_contrast > 0:
            boost_mask = self.create_bandpass_mask(freq_mag, contrast_frequency, 0.15)
            amplitude = amplitude * (1 + boost_mask * micro_contrast)
        
        # === 3. HIGH-FREQUENCY TEXTURE INJECTION ===
        # This is key: inject natural-looking texture into high frequencies
        if texture_injection > 0:
            amplitude = self.inject_texture(amplitude, freq_mag, texture_injection, 
                                           texture_frequency, channel.device)
        
        # === 4. REFERENCE-BASED TRANSFER (if provided) ===
        if reference_image is not None:
            amplitude = self.transfer_from_reference(
                amplitude, phase, reference_image, freq_mag, H, W
            )
        
        # Reconstruct
        fft_modified = amplitude * torch.exp(1j * phase)
        fft_unshifted = torch.fft.ifftshift(fft_modified)
        
        return torch.fft.ifft2(fft_unshifted).real

    def apply_slope_correction(self, amplitude, freq_mag, strength):
        """
        Correct frequency amplitude slope toward natural 1/f distribution.
        
        Real photos have amplitude ~ 1/f (pink noise).
        AI often has flatter high-frequency response.
        """
        # Avoid division by zero at DC
        safe_freq = freq_mag + 1e-4
        
        # Target 1/f slope (pink noise characteristic of natural images)
        # Current amplitude vs ideal 1/f relationship
        # We boost high frequencies that are "too weak" relative to 1/f
        
        # Calculate expected amplitude based on 1/f model
        # Normalize to not affect DC too much
        ideal_slope = 1.0 / safe_freq
        ideal_slope = ideal_slope / ideal_slope.max()
        
        # Only apply to high frequencies (above 0.1)
        high_freq_mask = torch.clamp((freq_mag - 0.1) / 0.4, 0, 1)
        
        # Current amplitude (normalized per-image)
        amp_normalized = amplitude / (amplitude.max() + 1e-8)
        
        # Calculate correction factor
        # Where amp is too low relative to ideal, boost it
        correction = ideal_slope / (amp_normalized + 1e-4)
        correction = torch.clamp(correction, 0.5, 2.0)  # Limit correction range
        
        # Apply correction only to high frequencies, scaled by strength
        correction_factor = 1 + (correction - 1) * high_freq_mask * strength * 0.3
        
        return amplitude * correction_factor

    def inject_texture(self, amplitude, freq_mag, strength, start_freq, device):
        """
        Inject natural-looking high-frequency texture.
        
        This adds subtle, structured noise to high frequencies that mimics
        real sensor texture. Key for fooling detectors on flat areas.
        """
        B, C, H, W = amplitude.shape
        
        # Create high-frequency mask (where to inject)
        hf_mask = torch.clamp((freq_mag - start_freq) / (0.5 - start_freq + 1e-4), 0, 1)
        hf_mask = hf_mask ** 0.5  # Softer transition
        
        # Generate structured texture (not just random noise)
        # Use multiple scales of noise for more natural appearance
        texture = torch.zeros_like(amplitude)
        
        # Fine texture (high frequency)
        fine_noise = torch.randn(B, C, H, W, device=device) * 0.5
        texture = texture + fine_noise * hf_mask
        
        # Medium texture (mid-high frequency)
        med_mask = torch.clamp((freq_mag - start_freq * 0.8) / 0.2, 0, 1) * \
                   torch.clamp((0.7 - freq_mag) / 0.2, 0, 1)
        med_noise = torch.randn(B, C, H, W, device=device) * 0.3
        texture = texture + med_noise * med_mask
        
        # Scale texture relative to existing amplitude
        # This ensures texture is proportional to image content
        amp_scale = amplitude.mean() * 0.1
        texture = texture * amp_scale * strength
        
        # Add texture to amplitude (always positive addition to amplitude)
        return amplitude + torch.abs(texture)

    def create_bandpass_mask(self, freq_mag, center, width):
        """Create a bandpass filter mask centered at given frequency."""
        return torch.exp(-((freq_mag - center) ** 2) / (2 * width ** 2))

    def transfer_from_reference(self, amplitude, phase, reference_image, freq_mag, H, W):
        """
        Transfer high-frequency characteristics from a reference image.
        
        This takes the texture/detail pattern from a real photo and
        applies it to the AI image's high frequencies.
        """
        ref = reference_image.permute(0, 3, 1, 2)
        
        # Resize reference if needed
        if ref.shape[2] != H or ref.shape[3] != W:
            ref = F.interpolate(ref, size=(H, W), mode='bicubic', align_corners=False)
        
        # Convert reference to luminance
        ref_lum = 0.299 * ref[:, 0:1] + 0.587 * ref[:, 1:2] + 0.114 * ref[:, 2:3]
        
        # Get reference spectrum
        ref_fft = torch.fft.fft2(ref_lum)
        ref_fft_shifted = torch.fft.fftshift(ref_fft)
        ref_amplitude = torch.abs(ref_fft_shifted)
        
        # Transfer only high frequencies (above 0.5)
        transfer_mask = torch.clamp((freq_mag - 0.4) / 0.2, 0, 1)
        
        # Blend: keep low-freq from original, take high-freq from reference
        blended = amplitude * (1 - transfer_mask * 0.5) + ref_amplitude * transfer_mask * 0.5
        
        return blended

    def match_statistics(self, result, original):
        """Match mean and std to prevent brightness/contrast shifts."""
        for c in range(result.shape[1]):
            orig_mean = original[:, c:c+1].mean()
            orig_std = original[:, c:c+1].std() + 1e-8
            result_mean = result[:, c:c+1].mean()
            result_std = result[:, c:c+1].std() + 1e-8
            
            result[:, c:c+1] = (result[:, c:c+1] - result_mean) / result_std * orig_std + orig_mean
        
        return result


# Node mappings
NODE_CLASS_MAPPINGS = {
    "FFTSurgeon": FFTSurgeon
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FFTSurgeon": "FFT Surgeon (Spectrum)"
}
