import cv2
import numpy as np
import onnxruntime
from typing import Optional

class FaceEnhancer:
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {
            'gfpgan_1.4': 'models/gfpgan_1.4.onnx',
            'gfpgan_1.3': 'models/gfpgan_1.3.onnx',
            'codeformer': 'models/codeformer.onnx',
            'gpen_bfr_512': 'models/gpen_bfr_512.onnx',
            'restoreformer_plus_plus': 'models/restoreformer_plus_plus.onnx'
        }
        self.current_model = 'gfpgan_1.4'
        self.face_enhancer = None
        self._load_model()
    
    def _load_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        model_path = self.models[self.current_model]
        try:
            self.face_enhancer = onnxruntime.InferenceSession(model_path, providers=providers)
        except:
            print(f"Model {model_path} not found, using fallback enhancement")
            self.face_enhancer = None
    
    def enhance(self, input_path, output_path, scale=2):
        input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        
        if self.face_enhancer is None:
            # Fallback enhancement
            enhanced_img = self._fallback_enhance(input_img)
        else:
            enhanced_img = self._model_enhance(input_img)
        
        # Apply additional enhancements
        enhanced_img = self._apply_post_processing(enhanced_img)
        
        cv2.imwrite(output_path, enhanced_img)
        return output_path
    
    def _model_enhance(self, image):
        faces = self._detect_faces(image)
        result_img = image.copy()
        
        for face in faces:
            enhanced_face = self._enhance_face_region(face, image)
            if enhanced_face is not None:
                x, y, w, h = face
                result_img[y:y+h, x:x+w] = enhanced_face
        
        return result_img
    
    def _enhance_face_region(self, face_bbox, image):
        x, y, w, h = face_bbox
        face_crop = image[y:y+h, x:x+w]
        
        # Resize to model input size
        model_size = self._get_model_size()
        face_crop = cv2.resize(face_crop, model_size)
        
        # Prepare input
        crop_vision_frame = self._prepare_crop_frame(face_crop)
        
        try:
            # Run inference
            inputs = {'input': crop_vision_frame}
            
            # Add weight input if model supports it
            if self._has_weight_input():
                inputs['weight'] = np.array([0.5]).astype(np.float64)
            
            outputs = self.face_enhancer.run(None, inputs)
            enhanced_crop = self._normalize_crop_frame(outputs[0][0])
            
            # Resize back to original face size
            enhanced_crop = cv2.resize(enhanced_crop, (w, h))
            return enhanced_crop
            
        except Exception as e:
            print(f"Face enhancement failed: {e}")
            return None
    
    def _detect_faces(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def _get_model_size(self):
        if self.current_model in ['gfpgan_1.4', 'gfpgan_1.3', 'codeformer', 'gpen_bfr_512', 'restoreformer_plus_plus']:
            return (512, 512)
        return (256, 256)
    
    def _prepare_crop_frame(self, crop_vision_frame):
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
        crop_vision_frame = np.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
        return crop_vision_frame
    
    def _normalize_crop_frame(self, crop_vision_frame):
        crop_vision_frame = np.clip(crop_vision_frame, -1, 1)
        crop_vision_frame = (crop_vision_frame + 1) / 2
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        crop_vision_frame = (crop_vision_frame * 255.0).round()
        crop_vision_frame = crop_vision_frame.astype(np.uint8)[:, :, ::-1]
        return crop_vision_frame
    
    def _has_weight_input(self):
        if self.face_enhancer is None:
            return False
        for input_info in self.face_enhancer.get_inputs():
            if input_info.name == 'weight':
                return True
        return False
    
    def _fallback_enhance(self, image):
        # Simple enhancement using OpenCV
        enhanced = cv2.bilateralFilter(image, 9, 75, 75)
        enhanced = cv2.addWeighted(image, 0.7, enhanced, 0.3, 0)
        return enhanced
    
    def _apply_post_processing(self, image):
        # Apply skin smoothing
        smooth = cv2.bilateralFilter(image, 15, 80, 80)
        image = cv2.addWeighted(image, 0.7, smooth, 0.3, 0)
        
        # Enhance sharpness
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        image = cv2.addWeighted(image, 0.8, sharpened, 0.2, 0)
        
        return image
    
    def set_model(self, model_name: str):
        if model_name in self.models:
            self.current_model = model_name
            self._load_model()
    
    def batch_enhance(self, input_dir, output_dir):
        import os
        os.makedirs(output_dir, exist_ok=True)
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"enhanced_{filename}")
                self.enhance(input_path, output_path)