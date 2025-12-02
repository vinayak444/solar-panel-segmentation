# Solar Panel Image Segmentation (U²-Net)

This repository provides a full pipeline for segmenting solar panels in RGB images using a fine-tuned U²-Net model.  
It supports local execution, Dockerized inference, and full AWS Lambda deployment using a container image.

---

# 🚀 1. Clone the Repository

Clone the repository from GitHub and move into the project directory:

    git clone https://github.com/vinayak444/solar-panel-segmentation.git
    cd solar-panel-segmentation

---

# 🧩 2. Local Python Environment Setup

## 2.1 Create a Virtual Environment

    python -m venv .venv

## 2.2 Activate the Environment

Windows:

    .\.venv\Scripts\activate

Linux / macOS:

    source .venv/bin/activate

## 2.3 Install Dependencies

    pip install --upgrade pip
    pip install -r requirements.txt

## 2.4 Add Model Weights (Required)

Place your trained U²-Net model inside:

    weights/u2net_multiclass.pt

This file is NOT included in git and must be added manually.

---

# 🖥️ 3. Run Segmentation Locally

## 3.1 Run on a Single Image

    python core.py --input image_097.jpg --output output_mask.png

This will:
- Load U²-Net from the weights folder  
- Preprocess the input  
- Generate a segmentation mask  
- Save it to `output_mask.png`  

## 3.2 Run on an Entire Folder

    python core.py --input-dir ./input_images --output-dir ./output_masks

This processes every file in `input_images/` and writes masks to `output_masks/`.

---

# 🐳 4. Docker (Local Inference)

## 4.1 Build Docker Image for Local Execution

    docker build -t solar-seg-local -f Dockerfile.local .

## 4.2 Run Container (API Mode)

    docker run --rm -p 8080:8080 solar-seg-local

Access the API:

    http://localhost:8080

## 4.3 Run Segmentation Inside Container (Script Mode)

Windows:

    docker run --rm -v %CD%:/app solar-seg-local ^
        python core.py --input image_097.jpg --output output_mask.png

Linux / macOS:

    docker run --rm -v $(pwd):/app solar-seg-local \
        python core.py --input image_097.jpg --output output_mask.png

---

# ☁️ 5. Build AWS Lambda-Compatible Docker Image

Use the provided Lambda Dockerfile:

    docker build -t solar-seg-lambda -f Dockerfile.lamda .

This builds an AWS Lambda–compatible image containing:
- The U²-Net model (from weights/)
- The inference logic (core.py)
- The Lambda handler (handler.py)

## 5.1 Optional: Test Lambda Image Locally

    docker run -p 9000:8080 solar-seg-lambda

Invoke it:

    curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
        -d "{\"test\":\"data\"}"

---

# 📦 6. Push Image to AWS ECR (Elastic Container Registry)

Replace **123456789012** with your AWS account ID.

## 6.1 Authenticate Docker to ECR

    aws ecr get-login-password --region eu-central-1 ^
        | docker login --username AWS --password-stdin 123456789012.dkr.ecr.eu-central-1.amazonaws.com

## 6.2 Tag the Docker Image

    docker tag solar-seg-lambda:latest \
        123456789012.dkr.ecr.eu-central-1.amazonaws.com/solar-segmentation:latest

## 6.3 Push Image to ECR

    docker push 123456789012.dkr.ecr.eu-central-1.amazonaws.com/solar-segmentation:latest

---

# 🟦 7. Deploy AWS Lambda Function (Container Image)

1. Open **AWS Console → Lambda → Create Function**
2. Choose **“Container Image”**
3. Select your ECR image:
   
       123456789012.dkr.ecr.eu-central-1.amazonaws.com/solar-segmentation:latest

4. Lambda automatically uses the handler defined in the Dockerfile:

       handler.lambda_handler

---

# 📥 8. Example Lambda Event Inputs

## 8.1 Base64 Image Input

    {
      "image_base64": "..."
    }

## 8.2 S3 File Input

    {
      "bucket": "my-bucket",
      "key": "image_097.jpg"
    }

---

# 🔌 9. API Gateway Integration (Optional)

Expose Lambda as an HTTP API:

    curl -X POST \
      -H "Content-Type: application/json" \
      -d "{\"image_base64\":\"<BASE64>\"}" \
      https://your-api-id.execute-api.eu-central-1.amazonaws.com/predict

---

# 📄 10. .gitignore

Recommended entries to avoid committing large files:

    weights/
    *.pt
    *.pth
    *.onnx
    __pycache__/
    *.pyc

---


This README section can be pasted directly into your repository.

