from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import json
import os

app = Flask(__name__)
CORS(app)

# =========================
# تحميل ملف المعلومات الغذائية
# =========================

JSON_PATH = "food101_health_info.json"

food_db = {}

try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        food_db = json.load(f)
    print(f"Loaded {len(food_db)} food items")
except Exception as e:
    print("Could not load nutrition database:", e)

# =========================
# تحميل الموديل مرة واحدة فقط
# =========================

MODEL_NAME = "nateraw/food"

print("Loading AI model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Model loaded successfully")

# =========================
# الصفحة الرئيسية
# =========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Food AI API Running",
        "status": "OK"
    })

# =========================
# التنبؤ
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        image = Image.open(file).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class_idx = outputs.logits.argmax(-1).item()

        label = model.config.id2label[predicted_class_idx]

        info = food_db.get(label.lower(), {})

        response = {
            "food_name": label.replace("_", " ").title(),
            "healthy": info.get("healthy", "N/A"),
            "calories": info.get("calories", "N/A"),
            "protein": info.get("protein", "N/A"),
            "carbs": info.get("carbs_g", "N/A"),
            "fat": info.get("fat_g", "N/A"),
            "warning": info.get("warning", ""),
            "benefits": info.get("benefits", []),
            "risks": info.get("risks", {}),
            "status": "Success"
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# =========================
# تشغيل السيرفر
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)