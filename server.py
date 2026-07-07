from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import pytesseract
import os

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        img_b64 = data["image"]
        img_bytes = base64.b64decode(img_b64)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        text = pytesseract.image_to_string(img, lang="dan")

        return jsonify({
            "success": True,
            "text": text
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

import os
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)

