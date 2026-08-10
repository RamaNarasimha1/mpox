"""
User Profile Routes
Handles user profile management endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

router = APIRouter(prefix="/user", tags=["user"])

# In-memory user profile store (replace with database in production)
user_profiles = {}


class UserProfile(BaseModel):
    """User profile data model"""
    name: str
    email: EmailStr
    avatar: Optional[str] = None
    joinDate: Optional[str] = None


class UserProfileResponse(BaseModel):
    """User profile response model"""
    name: str
    email: str
    avatar: Optional[str] = None
    joinDate: str


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile():
    """
    Get user profile
    
    Returns:
        UserProfileResponse: User profile data
    """
    # For demo purposes, return a default profile
    # In production, this would fetch from database based on authenticated user
    user_id = "default_user"
    
    if user_id in user_profiles:
        profile = user_profiles[user_id]
    else:
        profile = {
            "name": "Guest User",
            "email": "user@example.com",
            "avatar": None,
            "joinDate": datetime.now().isoformat()
        }
        user_profiles[user_id] = profile
    
    return profile


@router.put("/profile", response_model=UserProfileResponse)
async def update_profile(profile: UserProfile):
    """
    Update user profile
    
    Args:
        profile: Updated profile data
        
    Returns:
        UserProfileResponse: Updated user profile
    """
    user_id = "default_user"
    
    # Update profile
    updated_profile = {
        "name": profile.name,
        "email": profile.email,
        "avatar": profile.avatar,
        "joinDate": profile.joinDate or datetime.now().isoformat()
    }
    
    user_profiles[user_id] = updated_profile
    
    return updated_profile


@router.delete("/account")
async def delete_account():
    """
    Delete user account
    
    Returns:
        dict: Success message
    """
    user_id = "default_user"
    
    if user_id in user_profiles:
        del user_profiles[user_id]
    
    return {"message": "Account deleted successfully"}
