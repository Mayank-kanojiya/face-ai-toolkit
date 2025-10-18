import cv2
import numpy as np
import onnxruntime
from typing import List, Optional

class FaceSwapper:
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {
            'inswapper_128': 'models/inswapper_128.onnx',
            'simswap_256': 'models/simswap_256.onnx',
            'ghost_1_256': 'models/ghost_1_256.onnx'
        }
        self.current_model = 'inswapper_128'
        self.face_swapper = None
        self._load_model()
    
    def _load_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        model_path = self.models[self.current_model]
        self.face_swapper = onnxruntime.InferenceSession(model_path, providers=providers)
    
    def swap(self, source_path, target_path, output_path):
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)
        
        source_faces = self._detect_faces(source_img)
        target_faces = self._detect_faces(target_img)
        
        if not source_faces or not target_faces:
            raise ValueError("No faces detected in source or target image")
        
        result_img = target_img.copy()
        source_face = source_faces[0]
        
        for target_face in target_faces:
            result_img = self._swap_face(source_face, target_face, result_img)
        
        cv2.imwrite(output_path, result_img)
        return output_path
    
    def _detect_faces(self, image):
        # Simplified face detection - in production use proper face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return [{'bbox': face, 'embedding': self._extract_embedding(image, face)} for face in faces]
    
    def _extract_embedding(self, image, bbox):
        x, y, w, h = bbox
        face_crop = image[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (112, 112))
        face_crop = face_crop.astype(np.float32) / 255.0
        face_crop = np.transpose(face_crop, (2, 0, 1))
        return np.expand_dims(face_crop, axis=0)
    
    def _swap_face(self, source_face, target_face, image):
        # Simplified face swapping logic
        x, y, w, h = target_face['bbox']
        source_crop = cv2.resize(source_face['embedding'][0].transpose(1, 2, 0), (w, h))
        source_crop = (source_crop * 255).astype(np.uint8)
        
        # Simple blending
        mask = np.ones((h, w, 3), dtype=np.float32) * 0.8
        result = image.copy()
        result[y:y+h, x:x+w] = cv2.addWeighted(result[y:y+h, x:x+w], 0.2, source_crop, 0.8, 0)
        
        return result
    
    def swap_video(self, source_path, target_video_path, output_path):
        source_img = cv2.imread(source_path)
        source_faces = self._detect_faces(source_img)
        
        if not source_faces:
            raise ValueError("No face detected in source image")
        
        source_face = source_faces[0]
        cap = cv2.VideoCapture(target_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            target_faces = self._detect_faces(frame)
            result_frame = frame.copy()
            
            for target_face in target_faces:
                result_frame = self._swap_face(source_face, target_face, result_frame)
            
            out.write(result_frame)
        
        cap.release()
        out.release()
        return output_path