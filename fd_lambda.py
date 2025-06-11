# fd_lambda.py — Face Detection Lambda

import boto3
from facenet_pytorch import MTCNN
import os
import json
import base64
import numpy as np
from PIL import Image

class face_detection:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def face_detection_func(self, test_image_path, output_path):
        img = Image.open(test_image_path).convert("RGB")
        img = np.array(img)
        img = Image.fromarray(img)

        key = os.path.splitext(os.path.basename(test_image_path))[0].split(".")[0]

        face, prob = self.mtcnn(img, return_prob=True, save_path=None)

        if face != None:
            os.makedirs(output_path, exist_ok=True)

            face_img = face - face.min()
            face_img = face_img / face_img.max()
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()

            face_pil = Image.fromarray(face_img, mode="RGB")
            face_img_path = os.path.join(output_path, f"{key}_face.jpg")

            face_pil.save(face_img_path)
            return face_img_path
        else:
            print(f"No face is detected")
            return

sqs = boto3.client('sqs')
REQUEST_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/xxxxxxxxxxx/xxxxxxx-req-queue"

def image_to_base64(path: str) -> str:
    with open(path, 'rb') as img_file:
        encoded_bytes = base64.b64encode(img_file.read())
    return encoded_bytes.decode('utf-8')

def handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))
        content = body['content']
        req_id = body['request_id']
        fname = body.get('filename', '')
        
        img_bytes = base64.b64decode(content)
        face_img_path = f"/tmp/{req_id}_face.jpg"
        
        with open(face_img_path, 'wb') as f:
            f.write(img_bytes)
        
        face_output_path = f"/tmp/{req_id}_face_out.jpg"
        face_detection_obj = face_detection()
        face_img_path = face_detection_obj.face_detection_func(face_img_path, face_output_path)
        
        base64_file = image_to_base64(face_img_path)
        
        msg = json.dumps({
            'request_id': req_id,
            'face_img': base64_file,
            'filename': fname
        })
        
        sqs.send_message(
            QueueUrl=REQUEST_QUEUE_URL,
            MessageBody=msg
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({'request_id': req_id})
        }
    except Exception as e:
        print(f"[FD][Error] {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
