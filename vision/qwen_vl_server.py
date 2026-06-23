from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io

from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "prithivMLmods/Qwen2.5-VL-3B-Abliterated-Caption-GGUF"

app = FastAPI()

print("Loading Qwen2.5-VL-3B-Abliterated model...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, trust_remote_code=True)

class CaptionRequest(BaseModel):
    prompt: Optional[str] = "Describe this image in detail."

@app.post("/v1/images/captions")
async def caption_image(request: CaptionRequest = None, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prompt = (request.prompt if request and request.prompt else
              "Describe this image in detail.")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=128)
    caption = processor.batch_decode(out, skip_special_tokens=True)[0]
    return {"caption": caption}