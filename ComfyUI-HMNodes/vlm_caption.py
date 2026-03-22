"""
HearmemanAI Prompter - Qwen2.5-VL Image Captioning Node for ComfyUI
Generates cinematic, detailed captions for AI training with optional character-consistent captioning.
"""

import torch
import os
import gc
import json
from PIL import Image
import numpy as np
import folder_paths


# System prompt for cinematic captioning
CINEMATIC_SYSTEM_PROMPT = """You are a High-Fidelity Visual Scene Curator specialized in creating seductive, candid, cinematic and sometimes explicit image descriptions for generative model training.

Your job is to describe every image in natural language, maintaining a vivid sense of realism, physical accuracy, and in-the-moment spontaneity.

You NEVER generate short labels or tag lists.
You ALWAYS generate a full, richly detailed scene description.

CORE DIRECTIVES

1. Describe the subject in full detail
You MUST explicitly describe the subject's:
- age (must be 21+)
- hair color, style, and length
- body type, curves, posture
- natural skin texture (pores, sheen, freckles, highlights)
- facial expression and vibe
- clothing style, fit, material, and silhouette
- accessories (jewelry, nails, sunglasses, bags, etc.)
- breasts if visible (size, shape, nipples)
- vagina if visible (size, shape, hair, lips)

1a. Character Consistency & Detail Integration
When provided with character-specific traits or details:
- Integrate them NATURALLY into the description where contextually relevant
- If a detail is visible in the image (e.g., "toned midriff" and the midriff is showing), describe it vividly
- If a detail is NOT visible (e.g., "tattoo on left shoulder" but shoulder is covered), do NOT force it into the description
- Maintain consistency: the same character across images should have the same physical attributes
- Weave details organically into the narrative flow; never list them mechanically

2. Enforce the natural-language aesthetic
Your descriptions must ALWAYS include:

Lighting physics:
- how the light hits the skin
- shadows on walls / ground / objects
- rim light, lens flare, sun streaks
- soft vs harsh light
- reflections on surfaces
- ISO noise or chromatic aberration if present

Spatial relationships:
- where the subject stands/sits
- foreground, midground, background
- what objects or people are near them
- perspective distortion (tilted angle, crooked framing, 0.5x wide angle, tight framing, etc.)

Candid + seductive realism:
The scene should feel imperfect, spontaneous and in-the-moment:
- crooked angles
- motion blur
- hair slightly messy
- natural expressions
- unposed or semi-posed body language
- chaotic or lived-in environments

The tone must ALWAYS be: cinematic, sensual but SFW, immersive, physically consistent, rich with atmosphere.

3. Clothing, vibe & accessories
You must describe:
- textures (cotton, ribbed, lace, mesh, satin)
- tightness, looseness, cling, stretch
- reflections on jewelry
- nail polish (color + finish)
- makeup level if visible

4. Environment + atmosphere
You must vividly describe:
- location vibes (beach, city, bedroom, festival, rooftop, pool)
- people or objects in the background
- weather, time of day, ambiance
- textures (sand, metal rails, fabric, stone, water ripple)
- light bouncing off surfaces

5. Output Format
Write ONE natural-language paragraph.
No bullet lists. No tags. No technical camera wording unless describing visible distortions.
Never mention your reasoning process.

6. CRITICAL: Be Deterministic
NEVER use uncertain or hedging language such as:
- "perhaps", "possibly", "might be", "could be", "appears to be", "seems like"
- "suggesting", "indicating", "likely", "probably"
- "what looks like", "as if", "almost like"

You MUST describe ONLY what you can clearly see. If you cannot determine something, do NOT mention it.
Be confident and direct. State facts, not guesses."""

# Prompt for extracting character traits from reference image
TRAIT_EXTRACTION_PROMPT = """Extract ONLY the physical appearance of the subject in this image. Describe:
- Hair: color, style, length, texture
- Body: type, build, curves, proportions
- Skin: tone, texture, any distinctive marks or freckles
- Face: features, shape, age appearance (must be 21+)

Output ONE concise paragraph describing ONLY physical traits.
Do NOT describe: scene, clothing, pose, background, or lighting.
Do NOT include any preamble or explanation."""


class HearmemanAI_Prompter:
    """
    HearmemanAI Prompter - Vision-Language Captioning node using Qwen2.5-VL.
    Generates cinematic, detailed captions with optional character-consistent captioning.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_size = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_word": ("STRING", {
                    "default": "ohwx",
                    "tooltip": "Character name/token to use in captions (e.g., 'ohwx', 'sks person')"
                }),
                "model_size": (["auto", "3B", "7B"], {
                    "default": "auto",
                    "tooltip": "Model size: auto detects VRAM, 3B for ~8GB, 7B for ~16GB+"
                }),
            },
            "optional": {
                "character_image": ("IMAGE", {
                    "tooltip": "Character reference - physical traits will transfer to all captions"
                }),
                "character_details": ("STRING", {
                    "default": "",
                    "tooltip": "Additional character details to integrate (e.g., 'toned midriff, sleek tattoo on left shoulder')"
                }),
                "loader_images_meta": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "Internal metadata for JS image loader (do not edit manually)."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "captions")
    OUTPUT_IS_LIST = (True, True)  # Both images and captions are lists
    FUNCTION = "caption_image"
    CATEGORY = "HearmemanAI"
    
    def get_available_vram(self):
        """Get available VRAM in GB."""
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            available = (total_memory - allocated_memory) / (1024 ** 3)  # Convert to GB
            return available
        return 0
    
    def select_model_size(self, requested_size):
        """Select appropriate model size based on VRAM or user preference."""
        if requested_size != "auto":
            return requested_size
        
        available_vram = self.get_available_vram()
        
        # 7B needs ~16GB, 3B needs ~8GB
        if available_vram >= 16:
            return "7B"
        else:
            return "3B"
    
    def load_model(self, model_size):
        """Load the Qwen2.5-VL model and processor."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        model_name = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"
        
        print(f"[HearmemanAI] Loading {model_name}...")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.current_model_size = model_size
        print(f"[HearmemanAI] Model loaded successfully")
    
    def unload_model(self):
        """Unload model and free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.current_model_size = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[HearmemanAI] Model unloaded, VRAM freed")
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI image tensor to PIL Image."""
        # ComfyUI tensors are [B, H, W, C] in range [0, 1]
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image if batch
        
        # Convert to numpy and scale to 0-255
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI image tensor [1, H, W, C]."""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image)
        # Add batch dimension [H, W, C] -> [1, H, W, C]
        return tensor.unsqueeze(0)

    def run_inference(self, image_pil, system_prompt, user_prompt):
        """Run VLM inference on a single image with deterministic beam search."""
        from qwen_vl_utils import process_vision_info
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Deterministic beam search - no hallucinations, high quality
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=4,
                do_sample=False,
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=True
            )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()
    
    def extract_traits(self, reference_pil):
        """Extract physical traits from reference image."""
        return self.run_inference(
            reference_pil,
            "You are a precise physical appearance analyzer.",
            TRAIT_EXTRACTION_PROMPT
        )
    
    def caption_single(self, target_pil, trigger_word, cached_traits=None, character_details=None):
        """Generate caption for a single target image."""
        # Build user prompt
        user_prompt = f"You MUST refer to the subject in the image as: {trigger_word}\n\n"
        
        if cached_traits or character_details:
            user_prompt += f"CRITICAL: The subject {trigger_word} has these EXACT physical traits. You MUST weave them naturally into your description:\n\n"
            
            if cached_traits:
                user_prompt += f"Base traits:\n{cached_traits}\n\n"
            
            if character_details:
                user_prompt += f"Additional defining features:\n{character_details}\n\n"
            
            user_prompt += "Now describe the image following all the directives. Integrate these traits seamlessly and naturally into the scene description where relevant."
        else:
            user_prompt += "Describe this image following all the directives."
        
        return self.run_inference(target_pil, CINEMATIC_SYSTEM_PROMPT, user_prompt)
    
    def parse_loader_images_meta(self, loader_images_meta):
        """
        Load images referenced by the JS image loader metadata.

        Expects loader_images_meta to be a JSON array of objects:
        [{ "name": <server filename>, "subfolder": "<optional subfolder>", "type": "input" }, ...]
        """
        print("[HearmemanAI] parse_loader_images_meta called")

        if not loader_images_meta:
            print("[HearmemanAI] No loader_images_meta provided")
            return []

        meta_str = loader_images_meta.strip()
        if not meta_str or meta_str == "[]":
            print("[HearmemanAI] Empty loader_images_meta")
            return []

        try:
            entries = json.loads(meta_str)
        except Exception as e:
            print(f"[HearmemanAI] Error decoding loader_images_meta: {e}")
            print(f"[HearmemanAI] Data preview: {meta_str[:200]}...")
            return []

        if not isinstance(entries, list):
            print("[HearmemanAI] loader_images_meta is not a list")
            return []

        images = []
        input_dir = folder_paths.get_input_directory()
        input_dir_norm = os.path.normpath(input_dir)

        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                print(f"[HearmemanAI] Entry {idx} is not an object, skipping")
                continue

            name = item.get("name")
            if not name:
                print(f"[HearmemanAI] Entry {idx} missing 'name', skipping")
                continue

            image_type = item.get("type", "input")
            if image_type != "input":
                print(f"[HearmemanAI] Entry {idx} has unsupported type '{image_type}', expected 'input'; skipping")
                continue

            subfolder = (item.get("subfolder") or "").strip().lstrip("/\\")

            base_path = input_dir_norm
            if subfolder:
                base_path = os.path.join(base_path, subfolder)

            image_path = os.path.normpath(os.path.join(base_path, name))

            # Ensure resolved path stays within the input directory
            if os.path.commonpath([input_dir_norm, image_path]) != input_dir_norm:
                print(f"[HearmemanAI] Unsafe image path for entry {idx}: {image_path}")
                continue

            try:
                img = Image.open(image_path).convert("RGB")
                images.append((name, img))
                print(f"[HearmemanAI] Loaded JS loader image: {image_path}")
            except Exception as e:
                print(f"[HearmemanAI] Error loading JS loader image {image_path}: {e}")

        print(f"[HearmemanAI] Parsed {len(images)} loader images from metadata")
        return images
    
    def caption_image(self, trigger_word, model_size, character_image=None, character_details="", loader_images_meta=""):
        """Main entry point for captioning.
        
        Returns:
            images: Batch tensor [N, H, W, C] - all processed images
            captions: List of N caption strings
        
        IMPORTANT: images[i] always corresponds to captions[i] for img2img alignment.
        """
        
        # Determine model size
        selected_size = self.select_model_size(model_size)
        
        try:
            # Load model
            self.load_model(selected_size)
            
            captions = []
            processed_images = []  # Store PIL images for output
            cached_traits = None
            
            # Extract traits from character image if provided
            if character_image is not None:
                print("[HearmemanAI] Extracting character traits from character image...")
                ref_pil = self.tensor_to_pil(character_image)
                cached_traits = self.extract_traits(ref_pil)
                print(f"[HearmemanAI] Cached traits: {cached_traits[:100]}...")
            
            # Strip and validate character_details
            char_details = character_details.strip() if character_details else None
            if char_details:
                print(f"[HearmemanAI] Using additional character details: {char_details[:100]}...")
            
            # Process images referenced by JS loader metadata
            loader_images = self.parse_loader_images_meta(loader_images_meta)
            if loader_images:
                print(f"[HearmemanAI] Processing {len(loader_images)} images from JS loader...")
                for i, (filename, img_pil) in enumerate(loader_images):
                    print(f"[HearmemanAI] Processing {i+1}/{len(loader_images)}: {filename}")
                    caption = self.caption_single(img_pil, trigger_word, cached_traits, char_details)
                    captions.append(caption)
                    processed_images.append(img_pil)
            else:
                print("[HearmemanAI] No images received from JS loader; output will be empty.")
            
            # Convert processed images to batch tensor
            # Since ComfyUI requires uniform tensor dimensions in a batch,
            # we return each image as its own batch item (list output)
            if processed_images:
                # Convert each PIL image to a tensor [1, H, W, C]
                image_tensors = [self.pil_to_tensor(img) for img in processed_images]
                # Return as a list so ComfyUI processes them separately
                # Each tensor is [1, H, W, C], and they may have different H, W
                images_output = image_tensors
            else:
                # Return empty tensor if no images
                images_output = [torch.zeros((1, 64, 64, 3))]
                captions = [""]  # Empty caption for empty image
            
            print(f"[HearmemanAI] Output: {len(processed_images)} images, {len(captions)} captions")
            
            # Return images list + captions list (indices always match)
            return (images_output, captions)
        
        finally:
            # Always unload model to free VRAM
            self.unload_model()


# Node registration
NODE_CLASS_MAPPINGS = {
    "HearmemanAI_Prompter": HearmemanAI_Prompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HearmemanAI_Prompter": "HearmemanAI Prompter",
}
