import cv2
import numpy as np
import onnxruntime
from typing import Dict, Tuple

class AgeSelector:
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {
            'styleganex_age': {
                'path': 'models/styleganex_age.onnx',
                'templates': {
                    'target': 'ffhq_512',
                    'target_with_background': 'styleganex_384'
                },
                'sizes': {
                    'target': (256, 256),
                    'target_with_background': (384, 384)
                }
            }
        }
        self.current_model = 'styleganex_age'
        self.age_modifier = None
        self._load_model()
        
        # Age transformation parameters
        self.age_groups = {
            'child': (5, 12),
            'teen': (13, 19),
            'young_adult': (20, 35),
            'middle_aged': (36, 55),
            'senior': (56, 80)
        }
    
    def _load_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        model_info = self.models[self.current_model]
        try:
            self.age_modifier = onnxruntime.InferenceSession(model_info['path'], providers=providers)
        except:
            print(f"Model {model_info['path']} not found, using fallback age transformation")
            self.age_modifier = None
    
    def transform(self, input_path, output_path, target_age):
        image = cv2.imread(input_path)
        
        if self.age_modifier is None:
            # Fallback transformation
            transformed_image = self._fallback_transform(image, target_age)
        else:
            transformed_image = self._model_transform(image, target_age)
        
        cv2.imwrite(output_path, transformed_image)
        return output_path
    
    def _model_transform(self, image, target_age):
        faces = self._detect_faces(image)
        result_img = image.copy()
        
        for face in faces:
            transformed_face = self._modify_age_region(face, image, target_age)
            if transformed_face is not None:
                x, y, w, h = face
                result_img[y:y+h, x:x+w] = transformed_face
        
        return result_img
    
    def _modify_age_region(self, face_bbox, image, target_age):
        x, y, w, h = face_bbox
        face_crop = image[y:y+h, x:x+w]
        
        model_info = self.models[self.current_model]
        target_size = model_info['sizes']['target']
        background_size = model_info['sizes']['target_with_background']
        
        # Prepare face crops
        crop_vision_frame = cv2.resize(face_crop, target_size)
        extend_vision_frame = cv2.resize(face_crop, background_size)
        
        # Prepare inputs
        crop_prepared = self._prepare_vision_frame(crop_vision_frame)
        extend_prepared = self._prepare_vision_frame(extend_vision_frame)
        
        # Calculate age direction
        age_direction = np.array(np.interp(target_age, [0, 100], [2.5, -2.5])).astype(np.float32)
        
        try:
            # Run inference
            inputs = {
                'target': crop_prepared,
                'target_with_background': extend_prepared,
                'direction': age_direction
            }
            
            outputs = self.age_modifier.run(None, inputs)
            result_crop = self._normalize_extend_frame(outputs[0][0])
            
            # Resize back to original face size
            result_crop = cv2.resize(result_crop, (w, h))
            return result_crop
            
        except Exception as e:
            print(f"Age transformation failed: {e}")
            return None
    
    def _prepare_vision_frame(self, vision_frame):
        vision_frame = vision_frame[:, :, ::-1] / 255.0
        vision_frame = (vision_frame - 0.5) / 0.5
        vision_frame = np.expand_dims(vision_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
        return vision_frame
    
    def _normalize_extend_frame(self, extend_vision_frame):
        extend_vision_frame = np.clip(extend_vision_frame, -1, 1)
        extend_vision_frame = (extend_vision_frame + 1) / 2
        extend_vision_frame = extend_vision_frame.transpose(1, 2, 0).clip(0, 1)
        extend_vision_frame = (extend_vision_frame * 255.0)
        extend_vision_frame = extend_vision_frame.astype(np.uint8)[:, :, ::-1]
        return extend_vision_frame
    
    def _detect_faces(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def _fallback_transform(self, image, target_age):
        # Simple age transformation using image processing
        current_age_group = self.estimate_age_group(image)
        target_age_group = self.get_age_group_from_age(target_age)
        
        if self.is_aging_up(current_age_group, target_age_group):
            return self.age_up(image, target_age)
        else:
            return self.age_down(image, target_age)
    
    def estimate_age_group(self, image):
        # Simple age estimation based on facial features
        # In a real implementation, you'd use a trained age estimation model
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Analyze facial features for age estimation
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Simple heuristic based on face size and texture
            x, y, w, h = faces[0]
            face_region = gray[y:y+h, x:x+w]
            
            # Calculate texture variance (wrinkles indicator)
            texture_variance = np.var(face_region)
            
            if texture_variance < 100:
                return 'young_adult'
            elif texture_variance < 200:
                return 'middle_aged'
            else:
                return 'senior'
        
        return 'young_adult'  # Default
    
    def get_age_group_from_age(self, age):
        for group, (min_age, max_age) in self.age_groups.items():
            if min_age <= age <= max_age:
                return group
        return 'young_adult'
    
    def apply_age_transformation(self, image, current_group, target_group, target_age):
        if current_group == target_group:
            return image
        
        # Apply different transformations based on age direction
        if self.is_aging_up(current_group, target_group):
            return self.age_up(image, target_age)
        else:
            return self.age_down(image, target_age)
    
    def is_aging_up(self, current_group, target_group):
        age_order = ['child', 'teen', 'young_adult', 'middle_aged', 'senior']
        current_idx = age_order.index(current_group) if current_group in age_order else 2
        target_idx = age_order.index(target_group) if target_group in age_order else 2
        return target_idx > current_idx
    
    def age_up(self, image, target_age):
        # Add aging effects
        aged_image = image.copy()
        
        # Add wrinkles and texture
        aged_image = self.add_wrinkles(aged_image, intensity=min(target_age / 80.0, 1.0))
        
        # Adjust skin tone
        aged_image = self.adjust_skin_tone(aged_image, aging=True)
        
        # Add gray hair effect
        if target_age > 50:
            aged_image = self.add_gray_hair(aged_image, intensity=(target_age - 50) / 30.0)
        
        return aged_image
    
    def age_down(self, image, target_age):
        # Apply youth effects
        youthful_image = image.copy()
        
        # Smooth skin
        youthful_image = self.smooth_skin(youthful_image, intensity=0.7)
        
        # Brighten skin tone
        youthful_image = self.adjust_skin_tone(youthful_image, aging=False)
        
        # Enhance facial features
        youthful_image = self.enhance_youthful_features(youthful_image)
        
        return youthful_image
    
    def add_wrinkles(self, image, intensity=0.5):
        # Create wrinkle texture
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Generate wrinkle pattern
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        edges = cv2.filter2D(gray, -1, kernel)
        edges = np.clip(edges * intensity, 0, 255).astype(np.uint8)
        
        # Blend with original image
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(image, 1.0, edges_colored, 0.3 * intensity, 0)
        
        return result
    
    def adjust_skin_tone(self, image, aging=True):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        if aging:
            # Reduce saturation and brightness for aging
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 0.9)  # Reduce saturation
            hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 0.95)  # Slightly darken
        else:
            # Increase saturation and brightness for youth
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.1)  # Increase saturation
            hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.05)  # Slightly brighten
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    def add_gray_hair(self, image, intensity=0.5):
        # Simple gray hair effect
        gray_version = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_colored = cv2.cvtColor(gray_version, cv2.COLOR_GRAY2RGB)
        
        # Blend with original, focusing on hair regions (top portion)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        h, w = image.shape[:2]
        mask[:h//3, :] = intensity  # Top third of image (hair region)
        
        result = image.copy().astype(np.float32)
        for i in range(3):
            result[:,:,i] = result[:,:,i] * (1 - mask) + gray_colored[:,:,i] * mask
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def smooth_skin(self, image, intensity=0.5):
        # Bilateral filter for skin smoothing
        smooth = cv2.bilateralFilter(image, 15, 80, 80)
        return cv2.addWeighted(image, 1 - intensity, smooth, intensity, 0)
    
    def enhance_youthful_features(self, image):
        # Enhance eyes and lips for youthful appearance
        enhanced = image.copy()
        
        # Slight sharpening for feature definition
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        enhanced = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
        
        return enhanced