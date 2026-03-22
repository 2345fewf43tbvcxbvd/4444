import torch
import torch.nn.functional as F
import torch.fft
import io
from PIL import Image
import torchvision.transforms.functional as TF


class Analogizer:
    """
    Analog Pipeline (Pro) - A comprehensive post-processing node that simulates
    real camera physics to naturalize AI-generated images.
    
    Processing Order (physically accurate):
    1. FFT Cleaning (Anti-aliasing filter)
    2. Lens PSF Blur
    3. Chromatic Aberration (Lens distortion)
    4. Bayer Demosaic Artifacts
    5. Frequency Separation + Shot Noise (Sensor)
    6. JPEG Compression Artifacts
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # Frequency Domain
                "fft_cutoff": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 0.5, 
                    "step": 0.01,
                    "tooltip": "High-freq removal strength. 0=Off, 0.05-0.15=Subtle, 0.3+=Aggressive (causes blur)"
                }),
                "fft_order": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "tooltip": "Filter steepness. Higher=sharper cutoff"
                }),
                # Lens Simulation
                "lens_psf_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Lens point spread function blur. Simulates optical softness"
                }),
                "aberration_amount": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "Radial chromatic aberration strength in pixels"
                }),
                # Sensor Simulation
                "demosaic_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Bayer demosaicing artifact strength"
                }),
                "blur_radius": ("INT", {
                    "default": 9, 
                    "min": 1, 
                    "max": 64, 
                    "step": 2,
                    "tooltip": "Frequency separation kernel size"
                }),
                "noise_intensity": ("FLOAT", {
                    "default": 0.05, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Signal-dependent shot noise strength"
                }),
                "noise_correlation": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": "Spatial noise correlation scale. 0=Uncorrelated, 2-4=Sensor-like"
                }),
                # Compression
                "jpeg_quality": ("INT", {
                    "default": 100,
                    "min": 50,
                    "max": 100,
                    "step": 1,
                    "tooltip": "JPEG compression quality. 100=Off, 85-95=Subtle artifacts"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_analog_pipeline"
    CATEGORY = "SecretLab"

    def apply_analog_pipeline(self, image, fft_cutoff, fft_order, lens_psf_strength,
                               aberration_amount, demosaic_strength, blur_radius, 
                               noise_intensity, noise_correlation, jpeg_quality):
        # Setup: [B,H,W,C] -> [B,C,H,W]
        x = image.permute(0, 3, 1, 2).clone()
        
        # -----------------------------------------------------------
        # STEP 1: FFT CLEANING (Anti-Aliasing Filter)
        # Butterworth low-pass filter - removes AI grid artifacts
        # -----------------------------------------------------------
        if fft_cutoff > 0.0:
            x = self.apply_butterworth_filter(x, fft_cutoff, fft_order)

        # -----------------------------------------------------------
        # STEP 2: LENS PSF BLUR (Optical Softness)
        # Simulates real lens point spread function
        # -----------------------------------------------------------
        if lens_psf_strength > 0.0:
            x = self.apply_lens_psf(x, lens_psf_strength)

        # -----------------------------------------------------------
        # STEP 3: CHROMATIC ABERRATION (Lens Distortion)
        # Radial distortion - physically accurate
        # -----------------------------------------------------------
        if aberration_amount > 0.0:
            x = self.apply_chromatic_aberration(x, aberration_amount)

        # -----------------------------------------------------------
        # STEP 4: BAYER DEMOSAIC ARTIFACTS (Sensor Pattern)
        # Simulates color bleeding from Bayer filter interpolation
        # -----------------------------------------------------------
        if demosaic_strength > 0.0:
            x = self.add_demosaic_artifacts(x, demosaic_strength)

        # -----------------------------------------------------------
        # STEP 5: FREQUENCY SEPARATION + SHOT NOISE (Sensor Physics)
        # Signal-dependent noise injected into texture layer
        # -----------------------------------------------------------
        if noise_intensity > 0.0:
            x = self.apply_sensor_noise(x, blur_radius, noise_intensity, noise_correlation)

        # -----------------------------------------------------------
        # STEP 6: JPEG COMPRESSION (File Artifacts)
        # Real photos go through JPEG - AI images often don't
        # -----------------------------------------------------------
        if jpeg_quality < 100:
            x = self.apply_jpeg_artifacts(x, jpeg_quality)

        # Final cleanup
        result = torch.clamp(x, 0.0, 1.0)
        return (result.permute(0, 2, 3, 1),)

    # ===============================================================
    # HELPER FUNCTIONS
    # ===============================================================

    def apply_butterworth_filter(self, img, cutoff, order=2):
        """
        Butterworth high-frequency cut filter.
        Removes the highest frequencies where AI artifacts hide.
        
        Parameters:
            cutoff: 0.0 = no filtering, 0.5 = remove top 50% of frequencies
                    0.05-0.15 is typical for subtle AI artifact removal
        """
        B, C, H, W = img.shape
        
        # Transform to frequency domain
        fft = torch.fft.fft2(img)
        fft_shifted = torch.fft.fftshift(fft)
        
        # Create frequency grid (normalized, max ~0.707 at corners)
        fy = torch.linspace(-0.5, 0.5, H, device=img.device).view(-1, 1)
        fx = torch.linspace(-0.5, 0.5, W, device=img.device).view(1, -1)
        freq = torch.sqrt(fx**2 + fy**2)
        
        # Convert user's cutoff to Butterworth cutoff frequency
        # cutoff=0.0 → keep everything (actual_cutoff=0.5, no attenuation)
        # cutoff=0.1 → remove top 10% (actual_cutoff=0.45)
        # cutoff=0.5 → remove top 50% (actual_cutoff=0.25)
        max_freq = 0.5
        actual_cutoff = max_freq * (1.0 - cutoff)
        
        # Butterworth low-pass: smooth rolloff at cutoff frequency
        # Frequencies above actual_cutoff get progressively attenuated
        butterworth = 1.0 / (1.0 + (freq / (actual_cutoff + 1e-8))**(2 * order))
        butterworth = butterworth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Apply filter and transform back
        fft_filtered = fft_shifted * butterworth
        fft_ishift = torch.fft.ifftshift(fft_filtered)
        
        return torch.fft.ifft2(fft_ishift).real

    def apply_lens_psf(self, img, strength):
        """
        Apply realistic lens point spread function.
        Simulates slight hexagonal aperture blur with longitudinal CA.
        """
        B, C, H, W = img.shape
        kernel_size = 5
        
        # Create coordinate grid for kernel
        y = torch.linspace(-1, 1, kernel_size, device=img.device)
        x = torch.linspace(-1, 1, kernel_size, device=img.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Radial distance and angle
        r = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx)
        
        # Hexagonal aperture shape (6-blade simulation)
        hex_factor = 1 + 0.08 * torch.cos(6 * theta)
        kernel = torch.clamp(1.0 - r * hex_factor, 0, 1)
        kernel = kernel ** 2  # Softer falloff
        kernel = kernel / kernel.sum()
        
        # Create per-channel kernels with slight variation (longitudinal CA)
        kernel_r = kernel * (1 + 0.03 * strength)
        kernel_b = kernel * (1 - 0.03 * strength)
        
        kernel_r = (kernel_r / kernel_r.sum()).view(1, 1, kernel_size, kernel_size)
        kernel_g = kernel.view(1, 1, kernel_size, kernel_size)
        kernel_b = (kernel_b / kernel_b.sum()).view(1, 1, kernel_size, kernel_size)
        
        pad = kernel_size // 2
        r_ch = F.conv2d(img[:, 0:1], kernel_r, padding=pad)
        g_ch = F.conv2d(img[:, 1:2], kernel_g, padding=pad)
        b_ch = F.conv2d(img[:, 2:3], kernel_b, padding=pad)
        
        blurred = torch.cat([r_ch, g_ch, b_ch], dim=1)
        
        # Blend with original based on strength
        return img * (1 - strength) + blurred * strength

    def apply_chromatic_aberration(self, img, amount):
        """
        Radial chromatic aberration using grid_sample.
        Red channel pushed outward, blue pulled inward (barrel/pincushion).
        
        Features:
        - Physically accurate radial distortion
        - Edge falloff to prevent harsh color banding at borders
        - Smooth blending with original at extreme edges
        """
        B, C, H, W = img.shape
        
        # Create normalized coordinate grids [-1, 1]
        y = torch.linspace(-1, 1, H, device=img.device)
        x = torch.linspace(-1, 1, W, device=img.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Radial distance from center (0 at center, ~1.41 at corners)
        r = torch.sqrt(xx**2 + yy**2)
        
        # === EDGE FALLOFF MASK ===
        # Create a mask that's 1.0 in the center and fades near edges
        # This prevents harsh color banding at image borders
        edge_margin = 0.15  # Start fading at 85% from center
        max_r = 1.0  # Normalized edge distance
        
        # Smooth falloff: 1.0 in center, fades to 0.3 at edges
        # Using smoothstep-like curve for natural transition
        falloff_start = max_r - edge_margin
        edge_factor = torch.clamp((max_r - r) / edge_margin, 0.0, 1.0)
        edge_factor = edge_factor * edge_factor * (3 - 2 * edge_factor)  # Smoothstep
        edge_factor = 0.3 + 0.7 * edge_factor  # Range: 0.3 to 1.0
        
        # === RADIAL DISTORTION ===
        # Quadratic distortion scaled by edge falloff
        # Scaled to be subtle at reasonable amounts (1-5 pixels)
        scale_r = amount * 0.002 * edge_factor   # Red: push outward (barrel)
        scale_b = -amount * 0.002 * edge_factor  # Blue: pull inward (pincushion)
        
        # Apply radial distortion to coordinates
        r_grid_x = xx * (1 + scale_r * r**2)
        r_grid_y = yy * (1 + scale_r * r**2)
        b_grid_x = xx * (1 + scale_b * r**2)
        b_grid_y = yy * (1 + scale_b * r**2)
        
        # Stack into sampling grids [B, H, W, 2]
        r_grid = torch.stack([r_grid_x, r_grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        b_grid = torch.stack([b_grid_x, b_grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Sample each channel with its distortion
        r_ch = F.grid_sample(img[:, 0:1], r_grid, mode='bilinear', 
                             padding_mode='border', align_corners=True)
        g_ch = img[:, 1:2]  # Green stays centered
        b_ch = F.grid_sample(img[:, 2:3], b_grid, mode='bilinear', 
                             padding_mode='border', align_corners=True)
        
        result = torch.cat([r_ch, g_ch, b_ch], dim=1)
        
        # === EDGE BLEND ===
        # Additionally blend result with original at extreme edges
        # to completely eliminate any remaining edge artifacts
        edge_blend = torch.clamp((1.0 - r) / 0.1, 0.0, 1.0)  # 1.0 except at very edge
        edge_blend = edge_blend.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # At the very edge (r > 0.9), blend back toward original
        result = result * edge_blend + img * (1 - edge_blend)
        
        return result

    def add_demosaic_artifacts(self, img, strength):
        """
        Simulate Bayer demosaicing artifacts.
        Real cameras use Bayer color filter arrays - AI doesn't.
        Creates subtle color bleeding at edges.
        """
        B, C, H, W = img.shape
        
        # Create Bayer pattern masks
        # RGGB pattern: R at (0,0), G at (0,1) and (1,0), B at (1,1)
        checker_r = torch.zeros(H, W, device=img.device)
        checker_r[0::2, 0::2] = 1  # R pixels at even rows, even cols
        
        checker_b = torch.zeros(H, W, device=img.device)
        checker_b[1::2, 1::2] = 1  # B pixels at odd rows, odd cols
        
        # Green has checkerboard pattern (both G positions)
        checker_g = torch.zeros(H, W, device=img.device)
        checker_g[0::2, 1::2] = 1  # G at even row, odd col
        checker_g[1::2, 0::2] = 1  # G at odd row, even col
        
        # Create interpolation blur kernel (bilinear demosaic approximation)
        blur_kernel = torch.tensor([[1, 2, 1], 
                                    [2, 4, 2], 
                                    [1, 2, 1]], 
                                   device=img.device, dtype=torch.float32) / 16
        blur_kernel = blur_kernel.view(1, 1, 3, 3)
        
        # Blur each channel to simulate interpolation
        r_blur = F.conv2d(img[:, 0:1], blur_kernel, padding=1)
        g_blur = F.conv2d(img[:, 1:2], blur_kernel, padding=1)
        b_blur = F.conv2d(img[:, 2:3], blur_kernel, padding=1)
        
        # Interpolated mask (where original pixels would NOT exist in Bayer)
        interp_r = (1 - checker_r).unsqueeze(0).unsqueeze(0)
        interp_g = (1 - checker_g).unsqueeze(0).unsqueeze(0)
        interp_b = (1 - checker_b).unsqueeze(0).unsqueeze(0)
        
        # Mix: keep original at Bayer positions, blend blur at interpolated positions
        r_new = img[:, 0:1] * (1 - strength * interp_r) + r_blur * (strength * interp_r)
        g_new = img[:, 1:2] * (1 - strength * interp_g) + g_blur * (strength * interp_g)
        b_new = img[:, 2:3] * (1 - strength * interp_b) + b_blur * (strength * interp_b)
        
        return torch.cat([r_new, g_new, b_new], dim=1)

    def generate_correlated_noise(self, shape, correlation_scale, device):
        """
        Generate spatially correlated noise like real sensor readout.
        Real sensors have correlated noise from thermal gradients and amplifier crosstalk.
        """
        B, C, H, W = shape
        
        if correlation_scale <= 0:
            # Pure uncorrelated noise
            return torch.randn(B, C, H, W, device=device)
        
        # Generate at lower resolution for correlation
        scale = max(1, int(correlation_scale))
        small_h = max(4, H // scale)
        small_w = max(4, W // scale)
        
        # Low-frequency correlated component
        small_noise = torch.randn(B, C, small_h, small_w, device=device)
        correlated = F.interpolate(small_noise, size=(H, W), mode='bicubic', align_corners=False)
        
        # High-frequency uncorrelated component (temporal noise)
        uncorrelated = torch.randn(B, C, H, W, device=device)
        
        # Blend: more correlation = more low-freq component
        blend = min(0.8, correlation_scale / 8.0)
        return correlated * blend + uncorrelated * (1 - blend)

    def apply_sensor_noise(self, img, blur_radius, noise_intensity, noise_correlation):
        """
        Frequency separation with signal-dependent shot noise.
        Noise intensity follows Poisson statistics (sqrt of signal).
        """
        B, C, H, W = img.shape
        
        # Create Gaussian kernel for frequency separation
        kernel_size = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        sigma = kernel_size * 0.3
        k = self.get_gaussian_kernel(kernel_size, sigma).to(img.device)
        k = k.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        
        # Split into Low Frequency (color/tone) and High Frequency (texture)
        lf = F.conv2d(img, k, padding=kernel_size//2, groups=3)
        hf = img - lf + 0.5  # +0.5 offset for grey baseline
        
        # Generate noise (correlated or uncorrelated)
        raw_noise = self.generate_correlated_noise(hf.shape, noise_correlation, img.device)
        
        # Signal-dependent shot noise: variance proportional to signal level
        # Based on Poisson noise model: std = sqrt(signal)
        signal_factor = torch.sqrt(torch.clamp(lf, 0.01, 1.0))
        
        # Combine shot noise (signal-dependent) and read noise (constant floor)
        shot_noise = raw_noise * signal_factor * noise_intensity
        read_noise = raw_noise * noise_intensity * 0.15  # ~15% read noise floor
        final_noise = shot_noise + read_noise
        
        # Inject noise into high-frequency layer
        hf_noisy = hf + final_noise
        
        # Recombine
        return lf + (hf_noisy - 0.5)

    def apply_jpeg_artifacts(self, img, quality):
        """
        Apply real JPEG compression/decompression cycle.
        This adds authentic compression artifacts that detectors expect in real photos.
        """
        B, C, H, W = img.shape
        result = []
        
        for i in range(B):
            # Convert tensor to PIL Image
            frame = img[i].clamp(0, 1)
            pil_img = TF.to_pil_image(frame)
            
            # Compress to JPEG in memory buffer
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            
            # Decompress back
            pil_img = Image.open(buffer)
            
            # Convert back to tensor
            tensor = TF.to_tensor(pil_img)
            result.append(tensor)
        
        return torch.stack(result).to(img.device)

    def get_gaussian_kernel(self, size, sigma):
        """Generate 2D Gaussian kernel."""
        coords = torch.arange(size).float() - (size - 1) / 2
        grid = coords ** 2
        grid = grid.view(1, -1) + grid.view(-1, 1)
        kernel = torch.exp(-grid / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel


# Mapping for ComfyUI to load the node
NODE_CLASS_MAPPINGS = {
    "Analogizer": Analogizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Analogizer": "Analog Pipeline (Pro)"
}
