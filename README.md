# DermaVision AI

DermaVision AI is a full-stack skin-condition image-classification project. It combines a React dashboard, a FastAPI backend, PostgreSQL-backed user and analysis history, and an ensemble of deep-learning models with Grad-CAM visualizations.

> **Important:** This project is for academic/research demonstration only. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Features

- Image classification for Chickenpox, Measles, Monkeypox, and Normal cases
- Ensemble inference across DenseNet, EfficientNet, MobileNet, ResNet, ShuffleNet, SqueezeNet, and GhostNet models
- Grad-CAM visual explanations
- Authentication, user profiles, analysis history, statistics, and admin routes
- Batch image analysis and local/MinIO-compatible file storage
- Docker Compose development environment with PostgreSQL, Redis, and MinIO

## Results

### Reported test performance

The models were trained on a merged, de-duplicated dataset of **1,279 images** spanning Monkeypox, Chickenpox, Measles, and Normal classes. The data split was **70% training, 15% validation, and 15% test**.

| Approach | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| DenseNet121 | 91.26% | 88.09% | 87.98% | 87.84% |
| EfficientNet-B0 | 89.81% | 86.05% | 86.42% | 86.14% |
| MobileNetV3-Large | 90.78% | 89.22% | 87.01% | 87.93% |
| ResNet50 | 91.75% | 89.78% | 88.06% | 88.83% |
| ShuffleNetV2 | 88.35% | 86.29% | 85.12% | 85.64% |
| SqueezeNet1.1 | 87.38% | 86.77% | 83.29% | 84.02% |
| GhostNet-100 | 91.75% | 88.54% | 89.75% | 89.10% |
| Majority voting | 92.72% | 92.87% | 92.72% | 92.67% |
| **Dynamic entropy ensemble (removes two most-uncertain models)** | **93.69%** | **93.78%** | **93.69%** | **93.65%** |

The dynamic entropy ensemble was the best-performing approach. For each image, it measures model uncertainty, removes the two least-confident base models, and combines the remaining predictions. This improved accuracy by **1.97 percentage points** over majority voting and by **1.94 points** over the best individual model.

For every uploaded skin image, the web application returns the predicted class, confidence and ranked probabilities, a Grad-CAM heatmap, and a saved analysis record with image metadata and timestamp. These results are intended for academic evaluation only and are not clinical diagnostic claims.

The earlier TensorFlow experiments and their archived evaluation figure are retained in [`archive/legacy-tensorflow/`](archive/legacy-tensorflow/). They are historical research material; the current application uses the PyTorch ensemble described above.

## Tech stack

| Area | Technologies |
| --- | --- |
| Frontend | React 18, Vite, Tailwind CSS, Zustand, Axios |
| Backend | Python 3.10, FastAPI, SQLAlchemy, Pydantic |
| ML | PyTorch, Torchvision, TIMM, Albumentations, Grad-CAM |
| Services | PostgreSQL 15, Redis 7, MinIO, Docker Compose |

## Repository layout

```text
frontend/       React application
backend/app/    FastAPI routes, database models, authentication, inference, storage
backend/models/ Model weight files (not committed; see Model weights)
ml/             Training and ensemble experimentation scripts
tests/          Project tests
docker-compose.yml
```

## Run with Docker (recommended)

1. Install Docker Desktop and Docker Compose.
2. Copy the example environment file and set a strong `SECRET_KEY`:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Place the required model weight files in `backend/models/` (see below).
4. Start the stack:

   ```powershell
   docker compose up --build
   ```

5. Open the frontend at `http://localhost:5173`. The backend API documentation is at `http://localhost:8000/docs`.

## Run without Docker

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The backend needs PostgreSQL and Redis running, plus values in `.env` appropriate to your local services.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Model weights and datasets

Model weights and datasets are intentionally excluded from Git because they are large artifacts. The inference service expects these files in `backend/models/`:

```text
best_densenet121.pth
best_efficientnet_b0.pth
best_mobilenetv3_large.pth
best_resnet50.pth
best_shufflenetv2.pth
ghostnet_100_best.pth
squeezenet1_1_best.pth
```

Publish them separately using one of these approaches:

- **Git LFS** if collaborators should clone weights alongside the source.
- A GitHub Release, cloud storage link, or model registry for faster regular clones.

Do not commit patient images, uploads, database dumps, cloud credentials, or a real `.env` file.

## GitHub publishing checklist

- Commit the source code, Dockerfiles, dependency locks, training scripts, and this README.
- Keep `.env`, `node_modules`, uploaded images, databases, datasets, and model weights out of normal Git commits.
- Create a private repository if any data, weights, or project material has sharing restrictions.
- Before making the repository public, replace development credentials and review GitHub's secret scanning alerts.

## License

Add a license before public distribution. For an academic project, choose one that matches your institution's and dataset/model-license requirements.
