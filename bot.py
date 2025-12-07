import os
import asyncio
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort

from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ContentType

# TOKEN Render dagi Environment Variables dan olinadi
BOT_TOKEN = os.getenv("BOT_TOKEN")

# ONNX model fayli
MODEL_PATH = "thyroid_model.onnx"

# ONNX modelni yuklash
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

# Transformlar
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

CONF_THRESH = 80.0  # 80% dan past bo‘lsa — NOANIQ

def predict_pil(img):
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

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    file_info = await bot.get_file(message.photo[-1].file_id)
    img_path = "image.jpg"
    await bot.download_file(file_info.file_path, img_path)

    img = Image.open(img_path).convert("RGB")
    status, pred_class, benign_p, malignant_p, pred_prob = predict_pil(img)

    pred_upper = pred_class.upper()

    if status == "confident":
        text = (
            f"Sun'iy zakoning yakuniy xulosasi: *{pred_upper}* ({pred_prob:.2f}%)\n\n"
            f"Benign: {benign_p:.2f}%\n"
            f"Malignant: {malignant_p:.2f}%\n\n"
            "⚠️ Bu AI modeli, shifokor o'rnini bosa olmaydi."
        )
    else:
        text = (
            "Sun'iy zakoning yakuniy xulosasi: *NOANIQ* (ishonchsiz)\n\n"
            f"Benign: {benign_p:.2f}%\n"
            f"Malignant: {malignant_p:.2f}%\n\n"
            "⚠️ Model natijasi aniq emas. Bu AI modeli, shifokor o'rnini bosa olmaydi.\n"
            "Albatta tajribali shifokor bilan maslahat qiling."
        )

    await message.reply(text, parse_mode="Markdown")

@dp.message()
async def handle_other(message: types.Message):
    await message.reply("Iltimos, tiroid UTT rasmi yuboring 📷")

async def main():
    print("Bot Render’da ishga tushdi...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
