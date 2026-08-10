"""
Analysis History Routes
Handles analysis history CRUD operations from PostgreSQL database
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from app.database import get_db
from app.models import Case, User
from app.auth import get_current_user
from app.storage import storage

router = APIRouter(prefix="/analyses", tags=["analyses"])


class TopPrediction(BaseModel):
    """Individual prediction result"""
    class_name: Optional[str] = None  # Allow None for backwards compatibility
    confidence: float
    
    class ConfigDict:
        # Allow both 'class' and 'class_name' as field names
        populate_by_name = True
        
    # Support 'class' as alias for class_name
    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and 'class' in obj:
            obj['class_name'] = obj.pop('class')
        return super().model_validate(obj)


class AnalysisRecord(BaseModel):
    """Analysis record data model"""
    analysis_id: str
    predicted_class: str
    confidence: float
    top_predictions: List[TopPrediction]
    timestamp: str
    image_name: Optional[str] = None
    image_url: Optional[str] = None
    gradcam_url: Optional[str] = None
    created_at: Optional[str] = None  # For backwards compatibility with Dashboard


class AnalysisListResponse(BaseModel):
    """Paginated analysis list response"""
    items: List[AnalysisRecord]
    total: int
    page: int
    limit: int


@router.get("", response_model=AnalysisListResponse)
async def get_analyses(
    page: int = 1,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get paginated analysis history from database
    
    Args:
        page: Page number (1-indexed)
        limit: Items per page
        
    Returns:
        AnalysisListResponse: Paginated list of user's analyses
    """
    # Query total count for user
    total = db.query(Case).filter(Case.user_id == current_user.id).count()
    
    # Query paginated results (newest first)
    cases = db.query(Case).filter(
        Case.user_id == current_user.id
    ).order_by(
        Case.created_at.desc()
    ).offset((page - 1) * limit).limit(limit).all()
    
    # Convert to response format
    items = []
    for case in cases:
        items.append({
            "analysis_id": case.case_id,
            "predicted_class": case.predicted_label,
            "confidence": case.confidence,
            "top_predictions": [
                {"class_name": class_name, "confidence": conf}
                for class_name, conf in sorted(
                    case.probabilities.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ],
            "timestamp": case.created_at.isoformat(),
            "created_at": case.created_at.isoformat(),  # For Dashboard compatibility
            "image_name": case.original_filename,
            "image_url": storage.get_file_url(case.file_path) if case.file_path else None,
            "gradcam_url": storage.get_file_url(case.explanation_path) if case.explanation_path else None
        })
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit
    }


@router.get("/{analysis_id}", response_model=AnalysisRecord)
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get single analysis by ID from database
    
    Args:
        analysis_id: Case UUID
        
    Returns:
        AnalysisRecord: Analysis details
    """
    case = db.query(Case).filter(
        Case.case_id == analysis_id,
        Case.user_id == current_user.id
    ).first()
    
    if not case:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {
        "analysis_id": case.case_id,
        "predicted_class": case.predicted_label,
        "confidence": case.confidence,
        "top_predictions": [
            {"class_name": class_name, "confidence": conf}
            for class_name, conf in sorted(
                case.probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ],
        "timestamp": case.created_at.isoformat(),
        "image_name": case.original_filename,
        "image_url": storage.get_file_url(case.file_path) if case.file_path else None,
        "gradcam_url": storage.get_file_url(case.explanation_path) if case.explanation_path else None
    }


# Removed: Analysis creation now happens in predict endpoint


@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete analysis by ID from database
    
    Also deletes associated image files from storage.
    
    Args:
        analysis_id: Case UUID
        
    Returns:
        dict: Success message
    """
    case = db.query(Case).filter(
        Case.case_id == analysis_id,
        Case.user_id == current_user.id
    ).first()
    
    if not case:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Delete files from storage
    if case.file_path:
        storage.delete_file(case.file_path)
    if case.explanation_path:
        storage.delete_file(case.explanation_path)
    
    # Delete database record
    db.delete(case)
    db.commit()
    
    return {"message": "Analysis deleted successfully"}


@router.get("/{analysis_id}/export")
async def export_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export analysis as PDF from database
    
    Args:
        analysis_id: Case UUID
        
    Returns:
        StreamingResponse: PDF file
    """
    case = db.query(Case).filter(
        Case.case_id == analysis_id,
        Case.user_id == current_user.id
    ).first()
    
    if not case:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph(f"<b>DermaVision AI - Analysis Report</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Analysis details
    details = f"""
    <b>Analysis ID:</b> {case.case_id}<br/>
    <b>Predicted Condition:</b> {case.predicted_label}<br/>
    <b>Confidence:</b> {case.confidence * 100:.2f}%<br/>
    <b>Date:</b> {case.created_at.strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>Model Version:</b> {case.model_version}<br/>
    """
    
    if case.original_filename:
        details += f"<b>Image:</b> {case.original_filename}<br/>"
    
    story.append(Paragraph(details, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Top predictions
    story.append(Paragraph("<b>All Predictions:</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    for class_name, conf in sorted(case.probabilities.items(), key=lambda x: x[1], reverse=True):
        pred_text = f"{class_name}: {conf * 100:.2f}%"
        story.append(Paragraph(pred_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.pdf"}
    )
