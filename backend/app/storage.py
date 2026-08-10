"""File storage service for local filesystem and S3."""
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

settings = get_settings()


class StorageService:
    """Handles file storage operations."""
    
    def __init__(self):
        self.storage_type = settings.STORAGE_TYPE
        self.storage_root = Path(settings.STORAGE_ROOT)
        
        if self.storage_type == "local":
            # Create storage directories if they don't exist
            self.uploads_dir = self.storage_root / "uploads"
            self.gradcam_dir = self.storage_root / "gradcam"
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            self.gradcam_dir.mkdir(parents=True, exist_ok=True)
        elif self.storage_type == "s3":
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.S3_REGION
            )
            self.bucket_name = settings.S3_BUCKET
    
    def _generate_filename(self, original_filename: str, prefix: str = "") -> str:
        """Generate unique filename with timestamp and UUID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        ext = Path(original_filename).suffix.lower()
        if prefix:
            return f"{prefix}_{timestamp}_{unique_id}{ext}"
        return f"{timestamp}_{unique_id}{ext}"
    
    def save_image(
        self, 
        file_bytes: bytes, 
        original_filename: str, 
        user_id: int
    ) -> Tuple[str, int]:
        """
        Save uploaded image file.
        
        Args:
            file_bytes: Image file bytes
            original_filename: Original filename from upload
            user_id: User ID for organizing files
            
        Returns:
            Tuple of (file_path, file_size)
        """
        filename = self._generate_filename(original_filename, "img")
        file_size = len(file_bytes)
        
        if self.storage_type == "local":
            # Create user directory
            user_dir = self.uploads_dir / str(user_id)
            user_dir.mkdir(exist_ok=True)
            
            # Save file
            file_path = user_dir / filename
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            # Return relative path from storage root
            return str(file_path.relative_to(self.storage_root)), file_size
        
        elif self.storage_type == "s3":
            # S3 key with user folder structure
            s3_key = f"uploads/{user_id}/{filename}"
            
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=file_bytes,
                    ContentType="image/jpeg"
                )
                return f"s3://{self.bucket_name}/{s3_key}", file_size
            except ClientError as e:
                raise RuntimeError(f"Failed to upload to S3: {e}")
    
    def save_gradcam(
        self, 
        gradcam_bytes: bytes, 
        case_id: str
    ) -> str:
        """
        Save Grad-CAM visualization.
        
        Args:
            gradcam_bytes: Grad-CAM image bytes
            case_id: Case identifier
            
        Returns:
            file_path: Path to saved Grad-CAM image
        """
        filename = f"gradcam_{case_id}.png"
        
        if self.storage_type == "local":
            file_path = self.gradcam_dir / filename
            with open(file_path, "wb") as f:
                f.write(gradcam_bytes)
            return str(file_path.relative_to(self.storage_root))
        
        elif self.storage_type == "s3":
            s3_key = f"gradcam/{filename}"
            
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=gradcam_bytes,
                    ContentType="image/png"
                )
                return f"s3://{self.bucket_name}/{s3_key}"
            except ClientError as e:
                raise RuntimeError(f"Failed to upload Grad-CAM to S3: {e}")
    
    def get_file(self, file_path: str) -> Optional[bytes]:
        """
        Retrieve file bytes.
        
        Args:
            file_path: Path or S3 URI to file
            
        Returns:
            File bytes or None if not found
        """
        if file_path.startswith("s3://"):
            # Parse S3 URI
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
            
            try:
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                return response['Body'].read()
            except ClientError:
                return None
        else:
            # Local file
            full_path = self.storage_root / file_path
            if full_path.exists():
                with open(full_path, "rb") as f:
                    return f.read()
            return None
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete file.
        
        Args:
            file_path: Path or S3 URI to file
            
        Returns:
            True if deleted successfully
        """
        if file_path.startswith("s3://"):
            # Parse S3 URI
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
            
            try:
                self.s3_client.delete_object(Bucket=bucket, Key=key)
                return True
            except ClientError:
                return False
        else:
            # Local file
            full_path = self.storage_root / file_path
            if full_path.exists():
                full_path.unlink()
                return True
            return False
    
    def get_file_url(self, file_path: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for file access (S3 only).
        
        Args:
            file_path: Path or S3 URI to file
            expiration: URL expiration time in seconds
            
        Returns:
            Presigned URL or local path
        """
        if file_path.startswith("s3://"):
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
            
            try:
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': key},
                    ExpiresIn=expiration
                )
                return url
            except ClientError:
                return None
        else:
            # For local files, return the relative path
            # In production, you'd serve these through a web server
            return f"/files/{file_path}"


# Global storage instance
storage = StorageService()
