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
@app.route("/describe-image", methods=["POST"])
def describe_image():
  # Get image data from request (e.g., uploaded file)
  image_data = request.files["image"]

  # Define the prompt for describing the image
  prompt = """Please provide a detailed description of the image, focusing on the following elements:

  * Subject matter: What is the main focus of the image?
  * Setting: Where does the image take place?
  * Objects: What objects are present in the image?
  * Colors: What are the dominant colors?
  * Mood: What overall feeling or atmosphere does the image convey?

  Please be as descriptive as possible and include any relevant details."""

  # Use ollama to describe the image with the prompt
  res = ollama.chat(
      model="llava",
      messages=[
          {
              "role": "user",
              "content": prompt,  # Include the prompt in the message content
              "images": [image_data],
          }
      ]
  )

  # Return the generated description as JSON
  return {"description": res["message"]["content"]}
if __name__ == '__main__':
    app.run(debug=True)
