from ultralytics import YOLO
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import logging
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YOLO Anomaly Detection API",
    description="API for detecting anomalies using YOLOv8",
    version="1.0.0"
)

# Load your trained model
try:
    model = YOLO('model.pt')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@app.get("/")
async def root():
    return {"message": "YOLO Anomaly Detection API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await image.read()

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        original_h, original_w = img.shape[:2]

        # Resize for YOLO inference
        img_resized = cv2.resize(img, (640, 640))

        # Run inference
        results = model(img_resized, conf=0.2, iou=0.01, agnostic_nms=True)
        res = results[0]

        detections = []

        if res.boxes is not None and len(res.boxes) > 0:
            for box, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])

                # Map back to original image dimensions
                x1_orig = int(x1 * original_w / 640)
                y1_orig = int(y1 * original_h / 640)
                x2_orig = int(x2 * original_w / 640)
                y2_orig = int(y2 * original_h / 640)

                detections.append({
                    "x1": x1_orig,
                    "y1": y1_orig,
                    "x2": x2_orig,
                    "y2": y2_orig,
                    "confidence": float(conf),
                    "class": "anomaly"
                })

        logger.info(f"Processed image: {image.filename}, found {len(detections)} detections")
        return JSONResponse(content={
            "detections": detections,
            "image_info": {
                "width": original_w,
                "height": original_h,
                "filename": image.filename
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)