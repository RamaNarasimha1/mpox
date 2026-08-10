import random
import uuid
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from PIL import Image
from io import BytesIO

from app.ml_inference import predict as ml_predict, predict_ensemble
from app.database import get_db
from app.models import Case, User
from app.storage import storage
from app.auth import get_current_user

router = APIRouter(tags=["prediction"])

# the 4 skin conditions our model can detect
SKIN_CONDITIONS = [
    "Chickenpox",
    "Measles",
    "Monkeypox",
    "Normal",
]


@router.post("/predict")
async def predict_single(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    # main prediction endpoint - analyzes uploaded image
    
    # basic validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    contents = await file.read()
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    if len(contents) > 10 * 1024 * 1024:  # 10MB should be plenty
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    # run through our ensemble model - nst
    try:
        # get prediction with visualization
        prediction = predict_ensemble(contents, include_gradcam=True)
        
        case_id = str(uuid.uuid4())
        
        # save to storage
        file_path, file_size = storage.save_image(
            file_bytes=contents,
            original_filename=file.filename,
            user_id=current_user.id
        )
        
        # get image dimensions for metadata
        image = Image.open(BytesIO(contents))
        image_width, image_height = image.size
        
        # save the gradcam heatmap if we got one - rama
        explanation_path = None
        if "visualization" in prediction and "image" in prediction["visualization"]:
            gradcam_base64 = prediction["visualization"]["image"]
            if gradcam_base64.startswith("data:image"):
                gradcam_base64 = gradcam_base64.split(",")[1]
            gradcam_bytes = base64.b64decode(gradcam_base64)
            
            explanation_path = storage.save_gradcam(gradcam_bytes, case_id)
        
        # store everything in db
        case = Case(
            case_id=case_id,
            user_id=current_user.id,
            original_filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            image_width=image_width,
            image_height=image_height,
            predicted_label=prediction["predicted_class"],
            probabilities={pred["class"]: pred["confidence"] for pred in prediction["top_predictions"]},
            confidence=prediction["confidence"],
            model_version=prediction.get("model_version", "Multi-Model Ensemble v1.0"),
            explanation_path=explanation_path
        )
        
        db.add(case)
        db.commit()
        db.refresh(case)
        
        # Return prediction with case ID
        return {
            **prediction,
            "case_id": case_id,
            "image_name": file.filename,
            "image_size": file_size,
            "timestamp": case.created_at.isoformat(),
            "file_url": storage.get_file_url(file_path)
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    # batch upload - processes multiple images at once
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 10:  # limit to keep server happy
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "error": "Invalid file type - must be an image"
            })
            continue
        
        # Read and validate file
        try:
            contents = await file.read()
            
            if len(contents) == 0:
                results.append({
                    "filename": file.filename,
                    "error": "Empty file"
                })
                continue
            
            if len(contents) > 10 * 1024 * 1024:
                results.append({
                    "filename": file.filename,
                    "error": "File too large (max 10MB)"
                })
                continue
            
            # Use trained ML model for prediction with Grad-CAM
            try:
                prediction = predict_ensemble(contents, include_gradcam=True)
                
                # Generate unique case ID
                case_id = str(uuid.uuid4())
                
                # Save original image to storage
                file_path, file_size = storage.save_image(
                    file_bytes=contents,
                    original_filename=file.filename,
                    user_id=current_user.id
                )
                
                # Extract image dimensions
                image = Image.open(BytesIO(contents))
                image_width, image_height = image.size
                
                # Save Grad-CAM visualization if available
                explanation_path = None
                if "visualization" in prediction and "image" in prediction["visualization"]:
                    gradcam_base64 = prediction["visualization"]["image"]
                    if gradcam_base64.startswith("data:image"):
                        gradcam_base64 = gradcam_base64.split(",")[1]
                    gradcam_bytes = base64.b64decode(gradcam_base64)
                    explanation_path = storage.save_gradcam(gradcam_bytes, case_id)
                
                # Create case record in database
                case = Case(
                    case_id=case_id,
                    user_id=current_user.id,
                    original_filename=file.filename,
                    file_path=file_path,
                    file_size=file_size,
                    image_width=image_width,
                    image_height=image_height,
                    predicted_label=prediction["predicted_class"],
                    probabilities={pred["class"]: pred["confidence"] for pred in prediction["top_predictions"]},
                    confidence=prediction["confidence"],
                    model_version=prediction.get("model_version", "Multi-Model Ensemble v1.0"),
                    explanation_path=explanation_path
                )
                
                db.add(case)
                db.commit()
                db.refresh(case)
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "prediction": {
                        **prediction,
                        "case_id": case_id,
                        "image_name": file.filename,
                        "image_size": file_size,
                        "timestamp": case.created_at.isoformat(),
                        "file_url": storage.get_file_url(file_path)
                    }
                })
            except Exception as pred_error:
                db.rollback()
                results.append({
                    "filename": file.filename,
                    "error": f"Prediction failed: {str(pred_error)}"
                })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "total": len(files),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if "error" in r),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/conditions")
async def get_conditions() -> Dict[str, Any]:
    """
    Get list of detectable skin conditions.
    """
    return {
        "conditions": sorted(SKIN_CONDITIONS),
        "total": len(SKIN_CONDITIONS)
    }
