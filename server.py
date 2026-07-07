from paddleocr import PaddleOCR
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)

ocr = PaddleOCR(use_angle_cls=True, lang='latin')

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        img_b64 = data["image"]
        img_bytes = base64.b64decode(img_b64)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        result = ocr.ocr(img, cls=True)

        return jsonify({
            "success": True,
            "ocr": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

app.run(host="0.0.0.0", port=10000)
