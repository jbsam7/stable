from diffusers import StableVideoDiffusionPipeline
import torch
import os
from dotenv import load_dotenv
from PIL import Image
from huggingface_hub import login
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class SVDGenerator:
    def __init__(self, mode="img2vid", token=None):
        """
        Initialize the SVDGenerator with a mode.
        mode can be 'img2vid' or 'txt2vid'.
        """
        self.mode = mode
        self._authenticate(token)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Choose model based on mode
        self.model_id = (
            "stabilityai/stable-video-diffusion-img2vid-xt" if mode == "img2vid" 
            else "stabilityai/stable-video-diffusion-txt2vid"  # Example txt2vid model
        )

    def _authenticate(self, token=None):
        """Authenticate with HuggingFace."""
        if token:
            login(token=token)
        elif os.getenv('HF_TOKEN'):
            login(token=os.getenv('HF_TOKEN'))
        else:
            token_path = Path.home() / ".huggingface" / "token"
            if not token_path.exists():
                raise ValueError("Please provide token or set HF_TOKEN environment variable")    

    def load_model(self):
        """Load the model from Hugging Face."""
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None
        ).to(self.device)
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()  # Reduces memory usage
        self.pipe.to(self.device)

    def generate_from_text(self, prompt, num_frames=14, fps=7):
        """Generate video from a text prompt."""
        if not hasattr(self, 'pipe'):
            self.load_model()
        if self.mode != "txt2vid":
            raise ValueError("Text-to-video generation is only supported in 'txt2vid' mode.")
        return self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=50
        ).frames

    def generate_from_image(self, image_path, num_frames=14, fps=7):
        """Generate video from an image."""
        if not hasattr(self, 'pipe'):
            self.load_model()
        if self.mode != "img2vid":
            raise ValueError("Image-to-video generation is only supported in 'img2vid' mode.")
        image = Image.open(image_path)
        return self.pipe(
            image=image,
            num_frames=num_frames,
            num_inference_steps=50
        ).frames

if __name__ == "__main__":
    try:
        
        # Initialize for image-to-video generation
        generator_img = SVDGenerator(mode="img2vid")
        video_frames = generator_img.generate_from_image(
            image_path="draft_cover.jpg",  # Replace with your image path
            num_frames=14,
            fps=7
        )
        print(f"Generated {len(video_frames)} frames for image-to-video.")
        
    except Exception as e:
        print(f"Error during generation: {e}")
