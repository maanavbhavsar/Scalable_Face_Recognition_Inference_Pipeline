# fr_lambda.py — Face Recognition Lambda

import os
import json
import base64
import torch
import boto3
from PIL import Image
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageFont


class face_recognition:

    def face_recognition_func(self, model_path, model_wt_path, face_img_path):

        # Step 1: Load image as PIL
        face_pil = Image.open(face_img_path).convert("RGB")
        key = os.path.splitext(os.path.basename(face_img_path))[0].split(".")[0]

        # Step 2: Convert PIL to NumPy array (H, W, C) in range [0, 255]
        face_numpy = np.array(
            face_pil, dtype=np.float32
        )  # Convert to float for scaling

        # Step 3: Normalize values to [0,1] and transpose to (C, H, W)
        face_numpy /= 255.0  # Normalize to range [0,1]

        # Convert (H, W, C) → (C, H, W)
        face_numpy = np.transpose(face_numpy, (2, 0, 1))

        # Step 4: Convert NumPy to PyTorch tensor
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)

        saved_data = torch.load(model_wt_path)  # loading resnetV1_video_weights.pt

        self.resnet = torch.jit.load(
            model_path
        )  # this uses the model trace. resnetV1.pt

        if face_tensor != None:
            emb = self.resnet(
                face_tensor.unsqueeze(0)
            ).detach()  # detech is to make required gradient false
            embedding_list = saved_data[0]  # getting embedding data
            name_list = saved_data[1]  # getting list of names
            dist_list = (
                []
            )  # list of matched distances, minimum distance is used to identify the person

            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)

            idx_min = dist_list.index(min(dist_list))
            return name_list[idx_min]
        else:
            print(f"No face is detected")
            return


# One‑time model and data load per container
# model = InceptionResnetV1(pretrained='vggface2').eval()

# Load your "gallery" embeddings + labels
weights_path = os.path.join(
    os.environ.get("LAMBDA_TASK_ROOT", ""), "resnetV1_video_weights.pt"
)

model_path = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", ""), "resnetV1.pt")
# gallery = torch.load(weights_path, map_location='cpu')
# known_embeddings = gallery['embeddings']  # Tensor[N×512]
# known_labels     = gallery['labels']      # List[str] of length N

# # Pre‑processing pipeline
# pipeline = transforms.Compose([
#     transforms.Resize((160, 160)),
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])

sqs = boto3.client("sqs")
RESPONSE_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/xxxxxxxx/xxxxxx-resp-queue"


def process_record(record):
    try:
        body = json.loads(record["body"])
        req_id = body["request_id"]
        # face_id  = body.get('face_id', 0)
        face_b64 = body["face_img"]
        print(f"[FR] Processing request {req_id}...")
        # Decode & preprocess
        img_bytes = base64.b64decode(face_b64)

        # face_bytes = base64.b64decode(face_b64)
        face_img_path = f"/tmp/{req_id}_face.jpg"

        with open(face_img_path, "wb") as f:
            f.write(img_bytes)

        print(f"Face image saved to {face_img_path}")
        fr = face_recognition()
        # Use the provided face_recognition class
        identified_name = fr.face_recognition_func(
            model_path, weights_path, face_img_path
        )
        print(f"Identified name: {identified_name}")

        # Prepare response
        message = json.dumps(
            {
                "request_id": req_id,
                # 'face_id':    face_id,
                "result": identified_name,
            }
        )
        sqs.send_message(QueueUrl=RESPONSE_QUEUE_URL, MessageBody=message)
        print(f"[FR] Sent response for {message} to SQS.")
    except Exception as e:
        print(f"[FR][Error] {e}")


def handler(event, context):

    for record in event.get("Records", []):
        process_record(record)

    return {"statusCode": 200, "body": json.dumps({"processed": "success"})}
