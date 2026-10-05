from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
from PIL import Image
import io

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.predictor import predict_expression_from_photo

app = FastAPI(title="Facial Expression Recognition V2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allow local frontend development
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "V3-B Residual CNN"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
            image.load()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")
            
        try:
            result = predict_expression_from_photo(image)
        except ValueError as e:
            if "No face detected" in str(e):
                return JSONResponse(
                    status_code=422,
                    content={
                        "success": False,
                        "error": "NO_FACE_DETECTED",
                        "message": "No face was detected in the uploaded image."
                    }
                )
            else:
                raise HTTPException(status_code=400, detail=str(e))
                
        return {
            "success": True,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "bbox": result["bbox"]
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
