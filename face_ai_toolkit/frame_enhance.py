import cv2
import numpy as np
import onnxruntime
from typing import Tuple, List

class FrameEnhancer:
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {
            'real_esrgan_x4': {'path': 'models/real_esrgan_x4.onnx', 'scale': 4, 'size': (256, 16, 8)},
            'real_esrgan_x2': {'path': 'models/real_esrgan_x2.onnx', 'scale': 2, 'size': (256, 16, 8)},
            'ultra_sharp_x4': {'path': 'models/ultra_sharp_x4.onnx', 'scale': 4, 'size': (128, 8, 4)},
            'span_kendata_x4': {'path': 'models/span_kendata_x4.onnx', 'scale': 4, 'size': (128, 8, 4)},
            'clear_reality_x4': {'path': 'models/clear_reality_x4.onnx', 'scale': 4, 'size': (128, 8, 4)}
        }
        self.current_model = 'span_kendata_x4'
        self.frame_enhancer = None
        self._load_model()
    
    def _load_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        model_info = self.models[self.current_model]
        try:
            self.frame_enhancer = onnxruntime.InferenceSession(model_info['path'], providers=providers)
        except:
            print(f"Model {model_info['path']} not found, using fallback enhancement")
            self.frame_enhancer = None
    
    def enhance(self, input_path, output_path, scale=4):
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return self.enhance_video(input_path, output_path, scale)
        else:
            return self.enhance_image(input_path, output_path, scale)
    
    def enhance_image(self, input_path, output_path, scale=4):
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        
        if self.frame_enhancer is None:
            # Fallback enhancement
            enhanced_img = self._fallback_enhance(img, scale)
        else:
            enhanced_img = self._model_enhance(img)
        
        cv2.imwrite(output_path, enhanced_img)
        return output_path
    
    def enhance_video(self, input_path, output_path, scale=4):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        model_scale = self.models[self.current_model]['scale']
        output_width = width * model_scale
        output_height = height * model_scale
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.frame_enhancer is None:
                enhanced_frame = self._fallback_enhance(frame, model_scale)
            else:
                enhanced_frame = self._model_enhance(frame)
            
            out.write(enhanced_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        return output_path
    
    def _model_enhance(self, image):
        model_info = self.models[self.current_model]
        model_size = model_info['size']
        model_scale = model_info['scale']
        
        temp_height, temp_width = image.shape[:2]
        
        # Create tiles for processing
        tile_frames, pad_width, pad_height = self._create_tile_frames(image, model_size)
        
        enhanced_tiles = []
        for tile_frame in tile_frames:
            # Prepare tile
            prepared_tile = self._prepare_tile_frame(tile_frame)
            
            try:
                # Run inference
                outputs = self.frame_enhancer.run(None, {'input': prepared_tile})
                enhanced_tile = self._normalize_tile_frame(outputs[0])
                enhanced_tiles.append(enhanced_tile)
            except Exception as e:
                print(f"Tile enhancement failed: {e}")
                enhanced_tiles.append(tile_frame)
        
        # Merge tiles back
        enhanced_image = self._merge_tile_frames(
            enhanced_tiles, 
            temp_width * model_scale, 
            temp_height * model_scale,
            pad_width * model_scale,
            pad_height * model_scale,
            (model_size[0] * model_scale, model_size[1] * model_scale, model_size[2] * model_scale)
        )
        
        # Blend with original
        enhanced_image = self._blend_merge_frame(image, enhanced_image)
        
        return enhanced_image
    
    def _create_tile_frames(self, image, model_size):
        height, width = image.shape[:2]
        tile_size = model_size[0]
        
        # Calculate padding
        pad_width = (tile_size - width % tile_size) % tile_size
        pad_height = (tile_size - height % tile_size) % tile_size
        
        # Pad image
        padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_REFLECT)
        
        # Create tiles
        tiles = []
        for y in range(0, padded_image.shape[0], tile_size):
            for x in range(0, padded_image.shape[1], tile_size):
                tile = padded_image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
        
        return tiles, pad_width, pad_height
    
    def _prepare_tile_frame(self, tile_frame):
        tile_frame = np.expand_dims(tile_frame[:, :, ::-1], axis=0)
        tile_frame = tile_frame.transpose(0, 3, 1, 2)
        tile_frame = tile_frame.astype(np.float32) / 255.0
        return tile_frame
    
    def _normalize_tile_frame(self, tile_frame):
        tile_frame = tile_frame.transpose(0, 2, 3, 1).squeeze(0) * 255
        tile_frame = tile_frame.clip(0, 255).astype(np.uint8)[:, :, ::-1]
        return tile_frame
    
    def _merge_tile_frames(self, tiles, width, height, pad_width, pad_height, tile_size):
        # Simplified merge - in production use proper tile merging
        if tiles:
            merged = cv2.resize(tiles[0], (width, height))
            return merged[:height-pad_height, :width-pad_width] if pad_width > 0 or pad_height > 0 else merged
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def _blend_merge_frame(self, original, enhanced):
        # Resize original to match enhanced
        original_resized = cv2.resize(original, (enhanced.shape[1], enhanced.shape[0]))
        # Blend 80% enhanced, 20% original
        return cv2.addWeighted(original_resized, 0.2, enhanced, 0.8, 0)
    
    def _fallback_enhance(self, image, scale):
        # Simple upscaling with sharpening
        height, width = image.shape[:2]
        enhanced = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return enhanced
    
    def set_model(self, model_name: str):
        if model_name in self.models:
            self.current_model = model_name
            self._load_model()
    
    def denoise_frame(self, frame):
        return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)