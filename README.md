# Solar Panel Image Segmentation (U²-Net)

This repository contains a Python, Docker, and AWS Lambda–based segmentation pipeline for detecting solar panel regions in RGB images using a fine-tuned **U²-Net** model.

The project supports:

- ✔️ **Local segmentation** via `core.py`
- ✔️ **Serverless deployment** using **AWS Lambda container images**
- ✔️ **Dockerized inference**
- ✔️ **Batch processing** of images
- ✔️ A clean folder structure ready for research/production use

Large model files (e.g., `.pt`, `.pth`) are **not committed** to GitHub due to size limits.  
Place them manually inside the `weights/` directory.

---

# 📁 Repository Structure

├── core.py # Local entry point for segmentation
├── handler.py # AWS Lambda entry point
├── Dockerfile.lamda # Dockerfile for Lambda container deployment
├── Dockerfile.local # Dockerfile for local development (optional)
├── requirements.txt # Python dependencies
├── tools/ # Helper utilities for preprocessing/postprocessing
├── U2NET/ # U²-Net architecture and supporting code
├── weights/ # Model weights (ignored via .gitignore)
├── sample_images/ # Example input images (optional)
└── README.md


> ⚠️ **Note:**  
> Add your trained U²-Net model file to:
>
> ```
> weights/u2net_multiclass.pt
> ```
>
> This file is not included in Git.

---

# 🧩 1. Local Setup

## 1.1 Clone the Repository

```bash
git clone https://github.com/vinayak444/solar-panel-segmentation.git
cd solar-panel-segmentation

# 🧩 2. Local Setup Create a Python Virtual Environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
weights/u2net_multiclass.pt

pip install --upgrade pip
pip install -r requirements.txt
python core.py --input image_097.jpg --output output_mask.png
python core.py --input-dir ./input_images --output-dir ./output_masks
docker build -t solar-seg-local -f Dockerfile.local .
docker run --rm -p 8080:8080 solar-seg-local
docker run --rm -v %CD%:/app solar-seg-local \
  python core.py --input image_097.jpg --output output_mask.png
docker build -t solar-seg-lambda -f Dockerfile.lamda .
123456789012.dkr.ecr.eu-central-1.amazonaws.com/solar-segmentation
aws ecr get-login-password --region eu-central-1 \
 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.eu-central-1.amazonaws.com

docker tag solar-seg-lambda:latest 123456789012.dkr.ecr.eu-central-1.amazonaws.com/solar-segmentation:latest

docker push 123456789012.dkr.ecr.eu-central-1.amazonaws.com/solar-segmentation:latest
def lambda_handler(event, context):
    ...
{
  "image_base64": "..."
}
{
  "bucket": "my-bucket",
  "key": "image_097.jpg"
}
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<BASE64>"}' \
  https://your-api-id.execute-api.eu-central-1.amazonaws.com/predict


weights/
*.pt
*.pth
*.onnx
__pycache__/
*.pyc


