# Scalable Face Recognition Inference Pipeline

A serverless, scalable face recognition system built on AWS Lambda for real-time video frame processing.

## Overview

This project implements a two-stage face recognition pipeline:
1. **Face Detection**: Uses MTCNN to detect faces in video frames
2. **Face Recognition**: Uses InceptionResnetV1 (FaceNet) to identify detected faces

## Architecture

- **Face Detection Lambda**: Processes incoming video frames, detects faces, and sends them to SQS
- **Face Recognition Lambda**: Consumes from SQS, generates embeddings, and matches against a gallery
- **AWS SQS**: Message queue for asynchronous processing between stages

## Technologies

- AWS Lambda (Serverless)
- AWS SQS (Message Queue)
- PyTorch
- facenet-pytorch
- PIL/Pillow

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials and SQS queue URLs in the lambda functions

3. Build Docker image:
```bash
docker build -t face-recognition-lambda .
```

## Deployment

Deploy the Lambda functions using AWS SAM or Serverless Framework with the provided Dockerfile.

## Model Weights

The project requires pre-trained model weights:
- `resnetV1.pt`: Traced InceptionResnetV1 model
- `resnetV1_video_weights.pt`: Pre-computed embeddings and labels for face matching

These should be placed in the Lambda task root directory or configured via environment variables.
