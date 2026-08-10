#!/bin/bash

# Setup script for database and storage initialization

echo "==================================="
echo "DermaVision AI - Storage Setup"
echo "==================================="

# Create storage directories
echo "Creating storage directories..."
mkdir -p /app/uploads
mkdir -p /app/gradcam
mkdir -p /app/models

echo "✓ Storage directories created"

# Initialize database tables
echo "Initializing database tables..."
python init_db.py

echo "✓ Database tables created"

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Storage locations:"
echo "  - Images: /app/uploads/"
echo "  - Grad-CAM: /app/gradcam/"
echo "  - Models: /app/models/"
echo ""
echo "Database tables created:"
echo "  - users"
echo "  - cases"
echo "  - retraining_jobs"
echo "  - audit_logs"
echo ""
