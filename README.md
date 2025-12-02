## 🧩 1. Local Environment Setup

### 1.2 Create a Python Virtual Environment

Create a virtual environment in the project root:

    python -m venv .venv

### Activate the environment

Windows:

    .\.venv\Scripts\activate

Linux / macOS:

    source .venv/bin/activate

---

### 1.3 Install Dependencies

Upgrade pip and install all required Python packages:

    pip install --upgrade pip
    pip install -r requirements.txt

---

### 1.4 Add the Model Weights

Download or copy your trained U²-Net model and place it inside the weights folder:

    weights/u2net_multiclass.pt

This file is required for inference. It is not tracked in git and must be added manually.

---

## 🖥️ 2. Running Segmentation Locally

### 2.1 Run on a Single Image

Run the segmentation pipeline on one input image:

    python core.py --input image_097.jpg --output output_mask.png

What this does:

- Loads the U²-Net model from the weights directory
- Reads and preprocesses image_097.jpg
- Generates a segmentation mask
- Saves the result to output_mask.png

---

### 2.2 Run on a Folder of Images

Run segmentation on all images inside a folder:

    python core.py --input-dir ./input_images --output-dir ./output_masks

Explanation:

- Processes every image in input_images/
- Writes the corresponding mask for each image into output_masks/

---

## 🐳 3. Docker (Local Inference)

Docker allows a consistent environment across machines without installing dependencies directly on the host.

### 3.1 Build the Local Docker Image

Build a Docker image for local inference:

    docker build -t solar-seg-local -f Dockerfile.local .

Explanation:

- Builds an image named solar-seg-local
- Uses Dockerfile.local
- Installs dependencies and copies the project files into the container

---

### 3.2 Run the Container (HTTP API Mode)

If the container exposes an HTTP endpoint (for example with FastAPI or Flask):

    docker run --rm -p 8080:8080 solar-seg-local

Then you can open the following URL in a browser or send HTTP requests to it:

    http://localhost:8080

---

### 3.3 Run Segmentation Directly (Script Mode)

Run the segmentation script inside the container while mounting the current directory.

Windows:

    docker run --rm -v %CD%:/app solar-seg-local ^
      python core.py --input image_097.jpg --output output_mask.png

Linux / macOS:

    docker run --rm -v $(pwd):/app solar-seg-local \
      python core.py --input image_097.jpg --output output_mask.png

Explanation:

- Mounts your local project folder into /app inside the container
- Runs core.py using the Python environment inside the container
- Writes the output mask back to the mounted folder on your machine

---

## ☁️ 4. AWS Lambda Deployment (Container Image)

This project supports deploying the model as a serverless function using AWS Lambda with a container image.

### 4.1 Build Lambda-Compatible Docker Image

Build an image that follows the AWS Lambda container format:

    docker build -t solar-seg-lambda -f Dockerfile.lamda .

Explanation:

- Builds an image named solar-seg-lambda
- Uses Dockerfile.lamda
- Based on an AWS Lambda–compatible base image
- Includes the model and inference code

---

### 4.2 (Optional) Test Lambda Image Locally

Run the Lambda image locally using Docker:

    docker run -p 9000:8080 solar-seg-lambda

Invoke it with a test event:

    curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" ^
      -d "{\"test\": \"data\"}"

(or the same command without ^ on Linux/macOS, using a single line.)

This lets you verify that the Lambda handler is working before deploying to AWS.

---

## ☁️ 5. Push Image to AWS ECR

To use the image in AWS Lambda, push it to Amazon ECR (Elastic Container Registry). Replace 123456789012 with your own AWS account ID and adjust the region if needed.

### 5.1 Authenticate Docker to ECR

Log in Docker to your ECR registry:

    aws ecr get-login-password --region eu-central-1 ^
      | docker login --username AWS --password-stdin 123456789012.dkr.ecr.eu-central-1.amazonaws.com

---

### 5.2 Tag Your Image

Tag the local image with the ECR repository URI:

    docker tag solar-seg-lambda:latest 123456789012.dkr.ecr.eu-central-1.amazonaws.com/solar-segmentation:latest

---

### 5.3 Push the Image

Push the tagged image to ECR:

    docker push 123456789012.dkr.ecr.eu-central-1.amazonaws.com/solar-segmentation:latest

---

## 🟦 6. Deploy to AWS Lambda

After the image is available in ECR:

1. Open the AWS Console and go to AWS Lambda → Create function
2. Choose “Container image” as the function type
3. Select the repository and image tag from ECR (for example solar-segmentation:latest)
4. AWS Lambda will use the container’s entrypoint and handler, typically:

    handler.lambda_handler

(defined inside handler.py and referenced in the Dockerfile.)

---

### 6.1 Example Lambda Input Events

Base64 image input example:

    {
      "image_base64": "..."
    }

S3-based input example:

    {
      "bucket": "my-bucket",
      "key": "image_097.jpg"
    }

Your lambda_handler function should read this input, load the model, run segmentation, and either return the mask (for example as base64) or write it back to S3.

---

## 🔌 7. API Gateway (Optional HTTP Endpoint)

If you want a public HTTP endpoint, connect your Lambda function to API Gateway.

Example of calling the endpoint once it is configured:

    curl -X POST ^
      -H "Content-Type: application/json" ^
      -d "{\"image_base64\": \"<BASE64>\"}" ^
      https://your-api-id.execute-api.eu-central-1.amazonaws.com/predict

On Linux/macOS you can use the same command without line breaks or ^ characters.

---

## 📦 8. .gitignore

Use the following .gitignore rules to avoid committing large model files or temporary Python files:

    weights/
    *.pt
    *.pth
    *.onnx
    __pycache__/
    *.pyc
