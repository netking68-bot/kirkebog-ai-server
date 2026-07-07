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

        # ---------- PREPROCESSING ----------
        # 1) Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2) Resize (Tesseract elsker store billeder)
        scale = 2.0
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # 3) Fjern støj
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 4) Øg kontrast
        gray = cv2.equalizeHist(gray)

        # 5) Adaptive threshold (god til kirkebøger)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10
        )

        # ---------- TESSERACT CONFIG ----------
        config = (
            "--oem 1 "          # LSTM engine (bedst til håndskrift/tryk)
            "--psm 6 "          # Block of text
            "-l dan "           # Dansk OCR
        )

        text = pytesseract.image_to_string(thresh, config=config)

        return jsonify({
            "success": True,
            "text": text
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# ---------- RENDER PORT ----------
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)
