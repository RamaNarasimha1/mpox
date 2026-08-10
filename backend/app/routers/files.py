"""
File Serving Routes
Serves uploaded images and Grad-CAM visualizations
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.storage import storage

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{file_path:path}")
async def serve_file(file_path: str):
    """
    Serve a file from storage (images, grad-cam visualizations)
    
    Args:
        file_path: Relative path to the file in storage
        
    Returns:
        File contents with appropriate content-type
    """
    # Get file from storage
    file_bytes = storage.get_file(file_path)
    
    if file_bytes is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine content type based on file extension
    if file_path.endswith('.png'):
        content_type = "image/png"
    elif file_path.endswith(('.jpg', '.jpeg')):
        content_type = "image/jpeg"
    else:
        content_type = "application/octet-stream"
    
    return Response(content=file_bytes, media_type=content_type)
