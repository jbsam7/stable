from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch
import os

def generate_video(prompt1, prompt2, steps, scale, resolution):
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32  # Use mixed precision for GPUs

    pipeline = StableDiffusionWalkPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)

    # Parse resolution (e.g., "512x512")
    width, height = map(int, resolution.split("x"))

    # Generate the video
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    video_path = pipeline.walk(
        prompts=[prompt1, prompt2],
        seeds=[42, 1337],
        num_interpolation_steps=steps,
        guidance_scale=scale,
        height=height,
        width=width,
        output_dir=output_dir,
    )
    return video_path  # Return the path to the generated video


# API Client Function
def generate_video_via_api(prompt1, prompt2, steps, scale, resolution):
    from gradio_client import Client  # Import only when this function is called

    # Replace with the URL of your Gradio app (can be local or remote)
    client = Client("http://127.0.0.1:7860/")
    
    # Call the API endpoint
    result = client.predict(
        prompt1=prompt1,
        prompt2=prompt2,
        steps=steps,
        scale=scale,
        resolution=resolution,
        api_name="/predict"
    )
    print("API Result:", result)
    return result  # Returns the video path or output




if __name__ == "__main__":
    # Example inputs for both direct GPU processing and API usage
    prompt1 = "Lebron James sitting on a bench at a city park in a pink dress"
    prompt2 = "Lebron James standing up from sitting on a bench at a city park in a pink dress"
    steps = 100
    scale = 7.5
    resolution = "1024x1024"

    # Choice: Direct GPU Processing or API Communication
    mode = "gpu"  # Change to "api" if you want to use the API instead

    if mode == "gpu":
        print("Using local GPU to generate video...")
        video_path = generate_video(prompt1, prompt2, steps, scale, resolution)
        print(f"Video generated and saved at: {video_path}")
    elif mode == "api":
        print("Using Gradio API to generate video...")
        video_path = generate_video_via_api(prompt1, prompt2, steps, scale, resolution)
        print(f"Video generated via API at: {video_path}")
