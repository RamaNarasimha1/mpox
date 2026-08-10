"""
Statistics and Analytics Routes
Provides real-time statistics computed from PostgreSQL database
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any, List
from datetime import datetime, timedelta, date
from collections import Counter
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date

from app.database import get_db
from app.models import Case, User
from app.auth import get_current_user

router = APIRouter(prefix="/stats", tags=["statistics"])


@router.get("/dashboard")
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get dashboard statistics from database
    
    Returns:
        Dict containing:
        - totalAnalyses: Total number of user's analyses
        - todayAnalyses: Analyses from today
        - averageConfidence: Average confidence across all analyses
        - topConditions: Most frequently detected conditions
    """
    # Total analyses for user
    total_analyses = db.query(Case).filter(Case.user_id == current_user.id).count()
    
    if total_analyses == 0:
        return {
            "totalAnalyses": 0,
            "todayAnalyses": 0,
            "averageConfidence": 0.0,
            "topConditions": []
        }
    
    # Today's analyses
    today = datetime.now().date()
    today_analyses = db.query(Case).filter(
        Case.user_id == current_user.id,
        cast(Case.created_at, Date) == today
    ).count()
    
    # Average confidence
    avg_confidence_result = db.query(func.avg(Case.confidence)).filter(
        Case.user_id == current_user.id
    ).scalar()
    avg_confidence = float(avg_confidence_result) if avg_confidence_result else 0.0
    
    # Top conditions
    condition_counts = db.query(
        Case.predicted_label,
        func.count(Case.id).label('count')
    ).filter(
        Case.user_id == current_user.id
    ).group_by(
        Case.predicted_label
    ).order_by(
        func.count(Case.id).desc()
    ).limit(5).all()
    
    top_conditions = [
        {"name": label, "value": count}
        for label, count in condition_counts
    ]
    
    return {
        "totalAnalyses": total_analyses,
        "todayAnalyses": today_analyses,
        "averageConfidence": avg_confidence,
        "topConditions": top_conditions
    }


@router.get("/analytics")
async def get_analytics(
    period: str = "7d",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get time-series analytics data from database
    
    Args:
        period: Time period (7d, 30d, 90d)
        
    Returns:
        List of data points with date and count
    """
    # Parse period
    days = 7
    if period == "30d":
        days = 30
    elif period == "90d":
        days = 90
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)
    
    # Query count by date from database
    date_counts_query = db.query(
        cast(Case.created_at, Date).label('date'),
        func.count(Case.id).label('count')
    ).filter(
        Case.user_id == current_user.id,
        cast(Case.created_at, Date) >= start_date,
        cast(Case.created_at, Date) <= end_date
    ).group_by(
        cast(Case.created_at, Date)
    ).all()
    
    # Convert to dictionary
    date_counts = {str(date_obj): count for date_obj, count in date_counts_query}
    
    # Generate complete date range with zeros for missing days
    result = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.isoformat()
        result.append({
            "date": current_date.strftime("%b %d") if days > 7 else current_date.strftime("%a"),
            "count": date_counts.get(date_str, 0)
        })
        current_date += timedelta(days=1)
    
    return result


@router.get("/summary")
async def get_summary_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get comprehensive summary statistics from database
    
    Returns:
        Dict with various statistics about user's analyses
    """
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.8
    LOW_CONFIDENCE = 0.5
    
    # Total analyses
    total_analyses = db.query(Case).filter(Case.user_id == current_user.id).count()
    
    if total_analyses == 0:
        return {
            "totalAnalyses": 0,
            "uniqueConditions": 0,
            "averageConfidence": 0.0,
            "highConfidenceCount": 0,
            "lowConfidenceCount": 0,
            "conditionDistribution": {}
        }
    
    # Unique conditions
    unique_conditions = db.query(func.count(func.distinct(Case.predicted_label))).filter(
        Case.user_id == current_user.id
    ).scalar()
    
    # Average confidence
    avg_confidence = db.query(func.avg(Case.confidence)).filter(
        Case.user_id == current_user.id
    ).scalar()
    
    # High confidence count
    high_confidence_count = db.query(Case).filter(
        Case.user_id == current_user.id,
        Case.confidence >= HIGH_CONFIDENCE
    ).count()
    
    # Low confidence count
    low_confidence_count = db.query(Case).filter(
        Case.user_id == current_user.id,
        Case.confidence < LOW_CONFIDENCE
    ).count()
    
    # Condition distribution
    condition_dist_query = db.query(
        Case.predicted_label,
        func.count(Case.id)
    ).filter(
        Case.user_id == current_user.id
    ).group_by(Case.predicted_label).all()
    
    condition_distribution = {label: count for label, count in condition_dist_query}
    
    return {
        "totalAnalyses": total_analyses,
        "uniqueConditions": int(unique_conditions) if unique_conditions else 0,
        "averageConfidence": float(avg_confidence) if avg_confidence else 0.0,
        "highConfidenceCount": high_confidence_count,
        "lowConfidenceCount": low_confidence_count,
        "conditionDistribution": condition_distribution
    }
