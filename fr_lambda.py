# fr_lambda.py — Face Recognition Lambda

import os
import json
import base64
import torch
import boto3
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import numpy as np

class face_recognition:
    def face_recognition_func(self, model_path, model_wt_path, face_img_path):
        face_pil = Image.open(face_img_path).convert("RGB")
        
        face_numpy = np.array(face_pil, dtype=np.float32)
        face_numpy /= 255.0
        face_numpy = np.transpose(face_numpy, (2, 0, 1))
        
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)
        
        saved_data = torch.load(model_wt_path)
        self.resnet = torch.jit.load(model_path)
        
        if face_tensor != None:
            emb = self.resnet(face_tensor.unsqueeze(0)).detach()
            embedding_list = saved_data[0]
            name_list = saved_data[1]
            dist_list = []
            
            for idx, emb_db in enumerate(embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)
            
            idx_min = dist_list.index(min(dist_list))
            return name_list[idx_min]
        else:
            print(f"No face is detected")
            return

weights_path = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", ""), "resnetV1_video_weights.pt")
model_path = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", ""), "resnetV1.pt")

sqs = boto3.client("sqs")
RESPONSE_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/xxxxxxxx/xxxxxx-resp-queue"

def handler(event, context):
    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            req_id = body["request_id"]
            face_b64 = body["face_img"]
            
            img_bytes = base64.b64decode(face_b64)
            face_img_path = f"/tmp/{req_id}_face.jpg"
            
            with open(face_img_path, "wb") as f:
                f.write(img_bytes)
            
            fr = face_recognition()
            identified_name = fr.face_recognition_func(model_path, weights_path, face_img_path)
            
            message = json.dumps({
                "request_id": req_id,
                "result": identified_name,
            })
            
            sqs.send_message(QueueUrl=RESPONSE_QUEUE_URL, MessageBody=message)
        except Exception as e:
            print(f"[FR][Error] {e}")
    
    return {"statusCode": 200, "body": json.dumps({"processed": "success"})}

