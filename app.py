from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import cv2
import numpy as np
import torch
import io
from PIL import Image
app = Flask(__name__)

# Load models
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.enable_model_cpu_offload()

@app.route('/generate-image', methods=['POST'])
def generate_image():
    if 'prompt' not in request.form or 'image' not in request.files:
        return jsonify({"error": "Prompt and image file are required"}), 400

    prompt = request.form['prompt']
    image_file = request.files['image']

    try:
        # Load and process the image
        original_image = Image.open(image_file).convert("RGB")
        image = np.array(original_image)

        # Canny edge detection
        low_threshold = 100
        high_threshold = 200
        edges = cv2.Canny(image, low_threshold, high_threshold)
        edges_colored = np.stack([edges] * 3, axis=-1)  # Stack edges to create a 3-channel image
        canny_image = Image.fromarray(edges_colored)

        # Generate the new image
        negative_prompt = 'low quality, bad quality, sketches'
        generated_image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            controlnet_conditioning_scale=0.8,
        ).images[0]

        # Save the image to a BytesIO object
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name='generated_image.png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
