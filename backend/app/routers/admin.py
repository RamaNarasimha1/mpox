"""Admin routes."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from app.database import get_db
from app.models import User
from app.auth import get_current_user

router = APIRouter(prefix="/admin", tags=["admin"])


class UserListResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str
    created_at: str
    
    class Config:
        from_attributes = True


@router.get("/users", response_model=List[UserListResponse])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all users. Requires authentication."""
    # Get all users from database
    users = db.query(User).order_by(User.created_at.desc()).all()
    
    # Convert to response format
    return [
        UserListResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role.value,
            created_at=user.created_at.isoformat() if user.created_at else None
        )
        for user in users
    ]
