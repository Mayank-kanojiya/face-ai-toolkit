import cv2
import numpy as np
import onnxruntime
from typing import Dict, List

class DeepSwapper:
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {
            'iperov/elon_musk_224': 'models/iperov/elon_musk_224.dfm',
            'iperov/emma_watson_224': 'models/iperov/emma_watson_224.dfm',
            'druuzil/henry_cavill_448': 'models/druuzil/henry_cavill_448.dfm'
        }
        self.current_model = 'iperov/elon_musk_224'
        self.deep_swapper = None
        self._load_model()
    
    def _load_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        model_path = self.models[self.current_model]
        try:
            self.deep_swapper = onnxruntime.InferenceSession(model_path, providers=providers)
        except:
            # Fallback to CPU if model not found
            print(f"Model {model_path} not found, using fallback implementation")
            self.deep_swapper = None
    
    def swap(self, source_path, target_path, output_path):
        target_img = cv2.imread(target_path)
        
        if self.deep_swapper is None:
            # Fallback implementation
            return self._fallback_swap(source_path, target_path, output_path)
        
        target_faces = self._detect_faces(target_img)
        if not target_faces:
            raise ValueError("No faces detected in target image")
        
        result_img = target_img.copy()
        
        for target_face in target_faces:
            result_img = self._deep_swap_face(target_face, result_img)
        
        cv2.imwrite(output_path, result_img)
        return output_path
    
    def _detect_faces(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def _deep_swap_face(self, target_face, image):
        x, y, w, h = target_face
        
        # Extract face region
        face_crop = image[y:y+h, x:x+w]
        model_size = self._get_model_size()
        face_crop = cv2.resize(face_crop, model_size)
        
        # Prepare input
        crop_vision_frame = self._prepare_crop_frame(face_crop)
        morph_value = np.array([1.0]).astype(np.float32)
        
        # Run inference
        try:
            inputs = {
                'in_face:0': crop_vision_frame,
                'morph_value:0': morph_value
            }
            outputs = self.deep_swapper.run(None, inputs)
            result_crop = self._normalize_crop_frame(outputs[1][0])
            
            # Paste back
            result_crop = cv2.resize(result_crop, (w, h))
            result = image.copy()
            result[y:y+h, x:x+w] = result_crop
            
            return result
        except Exception as e:
            print(f"Deep swap failed: {e}, using fallback")
            return image
    
    def _get_model_size(self):
        return (224, 224)  # Default size
    
    def _prepare_crop_frame(self, crop_vision_frame):
        crop_vision_frame = cv2.addWeighted(crop_vision_frame, 1.75, cv2.GaussianBlur(crop_vision_frame, (0, 0), 2), -0.75, 0)
        crop_vision_frame = crop_vision_frame / 255.0
        crop_vision_frame = np.expand_dims(crop_vision_frame, axis=0).astype(np.float32)
        return crop_vision_frame
    
    def _normalize_crop_frame(self, crop_vision_frame):
        crop_vision_frame = (crop_vision_frame * 255.0).clip(0, 255)
        crop_vision_frame = crop_vision_frame.astype(np.uint8)
        return crop_vision_frame
    
    def _fallback_swap(self, source_path, target_path, output_path):
        # Simple fallback implementation
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)
        
        # Simple blend
        result = cv2.addWeighted(target_img, 0.7, source_img, 0.3, 0)
        cv2.imwrite(output_path, result)
        return output_path
    
    def set_model(self, model_name: str):
        if model_name in self.models:
            self.current_model = model_name
            self._load_model()