from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, 
    ForeignKey, JSON, Text, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


class UserRole(str, enum.Enum):
    # just regular user vs admin for now
    USER = "user"
    ADMIN = "admin"


class JobStatus(str, enum.Enum):
    # keeping track of ML training jobs
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)  # for banning users if needed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # link to their cases and audit trail
    cases = relationship("Case", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    __table_args__ = (
        Index("idx_user_email", "email"),
        Index("idx_user_role", "role"),
    )


class Case(Base):
    # each prediction/analysis creates one of these
    __tablename__ = "cases"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable for anonymous
    
    # info about the uploaded image
    original_filename = Column(String(255))
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer)
    image_width = Column(Integer)
    image_height = Column(Integer)
    
    # what the model predicted
    predicted_label = Column(String(128), index=True, nullable=False)
    probabilities = Column(JSON, nullable=False)  # stores all class probabilities
    confidence = Column(Float, nullable=False)
    model_version = Column(String(64), index=True, nullable=False)
    
    # path to the grad-cam visualization if we generated one
    explanation_path = Column(String(512))
    
    # let users correct our predictions - helps with retraining later!
    user_label = Column(String(128), index=True)
    user_labeled_at = Column(DateTime)
    
    # extra stuff we might want
    device_info = Column(JSON)
    exif_data = Column(JSON)  # image metadata
    consent_given = Column(Boolean, default=False, nullable=False)
    
    # timestamps for everything
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # back ref to user
    user = relationship("User", back_populates="cases")
    
    __table_args__ = (
        Index("idx_case_user_created", "user_id", "created_at"),
        Index("idx_case_predicted_label", "predicted_label"),
        Index("idx_case_confidence", "confidence"),
        Index("idx_case_model_version", "model_version"),
    )


class RetrainingJob(Base):
    """Model retraining job."""
    __tablename__ = "retraining_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    triggered_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    config = Column(JSON)  # Training config
    
    # Metrics
    train_samples = Column(Integer)
    val_samples = Column(Integer)
    metrics = Column(JSON)  # Accuracy, loss, per-class metrics
    
    # Results
    model_path = Column(String(512))
    new_version = Column(String(64))
    deployed = Column(Boolean, default=False)
    
    # Logs
    logs = Column(Text)
    error_message = Column(Text)
    
    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("idx_job_status", "status"),
        Index("idx_job_created", "created_at"),
    )


class AuditLog(Base):
    """Audit log for admin actions."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String(128), nullable=False, index=True)
    resource_type = Column(String(64))  # case, user, model, etc.
    resource_id = Column(String(64))
    details = Column(JSON)
    ip_address = Column(String(64))
    user_agent = Column(String(512))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index("idx_audit_user_action", "user_id", "action"),
        Index("idx_audit_created", "created_at"),
    )
