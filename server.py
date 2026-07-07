from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

app = Flask(__name__)

# Load Donut model (smallest version for Render Free)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        img_b64 = data["image"]
        img_bytes = base64.b64decode(img_b64)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prepare input
        pixel_values = processor(img, return_tensors="pt").pixel_values

        # Generate output
        output = model.generate(pixel_values, max_length=512)
        result = processor.tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({
            "success": True,
            "donut": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

app.run(host="0.0.0.0", port=10000)
