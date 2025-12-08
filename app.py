import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort
import gdown

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# ===== 1. Google Drive dan model zip'ni yuklash =====

# Sening link ID'ing: 1hP509plOi97mjEoR1PI6KLQL9MtJGU-k
GDRIVE_FILE_ID = "1hP509plOi97mjEoR1PI6KLQL9MtJGU-k"

MODEL_ZIP = Path("thyroid_model.zip")
MODEL_PATH = Path("thyroid_model.onnx")


def download_model_zip():
    """Google Drive'dan zip faylni yuklaydi."""
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print("Model zip yuklanmoqda Google Drive'dan...")
    gdown.download(url, str(MODEL_ZIP), quiet=False)


def ensure_model_file():
    """Agar ONNX model yo'q bo'lsa, zipni Drive'dan yuklab, ichidan ochadi."""
    if MODEL_PATH.exists():
        print("ONNX model topildi, qayta yuklash shart emas.")
        return

    if not MODEL_ZIP.exists():
        download_model_zip()

    print("Zipdan ONNX fayllar ochilmoqda...")
    with zipfile.ZipFile(MODEL_ZIP, "r") as z:
        z.extractall(".")  # ichidan thyroid_model.onnx (+ .data) chiqadi

    if not MODEL_PATH.exists():
        raise FileNotFoundError("thyroid_model.onnx zip ichidan topilmadi.")


# ===== 2. Modelni tayyorlab, ONNX sessiyani ishga tushiramiz =====
ensure_model_file()

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
print("ONNX model yuklandi. Input name:", INPUT_NAME)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

CONF_THRESH = 80.0  # % dan yuqori bo'lsa "ishonchli"


def predict_pil(img: Image.Image):
    x = transform(img).unsqueeze(0).numpy().astype(np.float32)
    outputs = session.run(None, {INPUT_NAME: x})
    logits = outputs[0][0]
    probs = softmax(logits) * 100.0

    benign_p = float(probs[0])
    malignant_p = float(probs[1])

    if benign_p >= malignant_p:
        pred_class = "Benign"
        pred_prob = benign_p
    else:
        pred_class = "Malignant"
        pred_prob = malignant_p

    status = "confident" if pred_prob >= CONF_THRESH else "uncertain"
    return status, pred_class, benign_p, malignant_p, pred_prob


# ===== 3. Flask web ilova =====
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    details = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            result = "Rasm tanlanmadi."
        else:
            filename = secure_filename(file.filename)
            if filename == "":
                filename = "upload.jpg"
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            img = Image.open(save_path).convert("RGB")
            status, pred_class, benign_p, malignant_p, pred_prob = predict_pil(img)

            pred_upper = pred_class.upper()

            if status == "confident":
                result = f"Sun'iy zakoning yakuniy xulosasi: {pred_upper} ({pred_prob:.2f}%)"
            else:
                result = "Sun'iy zakoning yakuniy xulosasi: NOANIQ (ishonchsiz)"

            details = f"Benign: {benign_p:.2f}%, Malignant: {malignant_p:.2f}%"

    return render_template("index.html", result=result, details=details)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
