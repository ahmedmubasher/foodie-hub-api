import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import TFAutoModelForImageClassification, AutoImageProcessor
import tensorflow as tf
from PIL import Image
import json

# اسم ملف البيانات
health_info_file = r"C:\food-101\images\food101_health_info.json"
health_info = {} 
print("Current folder:", os.getcwd())
print("Looking for:", os.path.abspath(health_info_file))
print("Exists:", os.path.exists(health_info_file))
# محاولة تحميل ملف الـ JSON
try:
    with open(health_info_file, encoding="utf-8") as f:
        health_info = json.load(f) 
    print(f"Successfully loaded {len(health_info)} food items from {health_info_file}")
except FileNotFoundError:
    print(f"Warning: File {health_info_file} not found → nutritional analysis will not be shown")
except Exception as e:
    print(f"Error reading health data file: {e}")

model_name = "nateraw/food"
image_path = r"C:\food-101\images\sushi\2357.jpg"

print(os.getcwd())
print("--- Analysis in progress, please wait a few seconds ---")

try:
    if not os.path.exists(image_path):
        raise FileNotFoundError

    # تحميل النموذج والبرنامج المساعد
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = TFAutoModelForImageClassification.from_pretrained(model_name, from_pt=True)

    # معالجة الصورة
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="tf")

    # التنبؤ بنوع الطعام
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = tf.math.argmax(logits, axis=-1).numpy()[0]
    label = model.config.id2label[predicted_class_idx]

    print("\n******************************")
    print(f"[IDENTIFIED AS] {label}")

    # تحويل اسم الطعام إلى حروف صغيرة للتأكد من مطابقتها داخل ملف الـ JSON
    json_lookup_key = label.lower()

    if json_lookup_key in health_info:
        info = health_info[json_lookup_key]
        print("\n[Health Grade:")
        print(f"→ {info.get('healthy', 'Assessment not available')}\n")
        
        print("[APPROXIMATE NUTRITIONAL VALUES]:")  
        print("\n******************************")
        print(f" • Calories : {info.get('calories', '?')}")
        print(f" • Protein : {info.get('protein', '?')}")
        print(f" • Carbohydrates : {info.get('carbs_g', '?')} ")
        print(f" • Fat : {info.get('fat_g', '?')} ")
        
        warning_msg = info.get('warning')
        if warning_msg:
            print(f"\n⚠️ Warning: {warning_msg}")
            
        print("******************************")

        if "benefits" in info:
            print("Main Benefits (الفوائد الرئيسية):")
            for b in info["benefits"]:
                print(f" - {b}")
        
        if "risks" in info:
            print("\nRisks / Warnings by Health Condition:")
            for disease, note in info["risks"].items():
                print(f" • {disease}: {note}")
    else:
        print(f"\nNote: No nutritional analysis is currently available for '{label}'.")
        print("تأكد من إضافة هذا الاسم بنفس الحروف الصغيرة داخل ملف food101_health_info.json")

except FileNotFoundError:
    print(f"\nError: Image not found → {image_path}")
except Exception as e:
    print(f"\nUnexpected error during analysis: {e}")
    model.save('food101_model.keras')