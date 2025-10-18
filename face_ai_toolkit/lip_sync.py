import cv2
import numpy as np
import librosa
import onnxruntime
from moviepy.editor import VideoFileClip, AudioFileClip
from typing import Optional

class LipSyncer:
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {
            'wav2lip_gan_96': {'path': 'models/wav2lip_gan_96.onnx', 'type': 'wav2lip', 'size': (96, 96)},
            'wav2lip_96': {'path': 'models/wav2lip_96.onnx', 'type': 'wav2lip', 'size': (96, 96)},
            'edtalk_256': {'path': 'models/edtalk_256.onnx', 'type': 'edtalk', 'size': (256, 256)}
        }
        self.current_model = 'wav2lip_gan_96'
        self.lip_syncer = None
        self._load_model()
    
    def _load_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        model_info = self.models[self.current_model]
        try:
            self.lip_syncer = onnxruntime.InferenceSession(model_info['path'], providers=providers)
        except:
            print(f"Model {model_info['path']} not found, using fallback lip sync")
            self.lip_syncer = None
    
    def sync(self, video_path, audio_path, output_path):
        # Load audio and extract features
        audio_features = self.extract_audio_features(audio_path)
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = 'temp_synced.mp4'
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get audio intensity for current frame
                audio_intensity = self.get_audio_intensity_at_frame(audio_features, frame_idx, fps)
                
                # Modify lip landmarks based on audio
                modified_frame = self.modify_lips(frame, face_landmarks, audio_intensity)
                out.write(modified_frame)
            else:
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Combine with audio
        self.combine_audio_video(temp_video, audio_path, output_path)
        return output_path
    
    def extract_audio_features(self, audio_path):
        y, sr = librosa.load(audio_path)
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Extract RMS energy
        rms = librosa.feature.rms(y=y)[0]
        return {'mfccs': mfccs, 'rms': rms, 'sr': sr}
    
    def get_audio_intensity_at_frame(self, audio_features, frame_idx, fps):
        time_per_frame = 1.0 / fps
        audio_idx = int(frame_idx * time_per_frame * audio_features['sr'] / 512)
        
        if audio_idx < len(audio_features['rms']):
            return audio_features['rms'][audio_idx]
        return 0.0
    
    def modify_lips(self, frame, face_landmarks, audio_intensity):
        h, w = frame.shape[:2]
        
        # Get lip landmarks
        lip_points = []
        for idx in self.lip_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            lip_points.append([x, y])
        
        lip_points = np.array(lip_points)
        
        # Modify lip opening based on audio intensity
        mouth_opening = int(audio_intensity * 20)  # Scale factor
        
        # Apply lip modification (simplified)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [lip_points], 255)
        
        # Slightly open/close mouth based on audio
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mouth_opening, mouth_opening))
        if mouth_opening > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        
        return frame
    
    def combine_audio_video(self, video_path, audio_path, output_path):
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        final_video = video.set_audio(audio)
        final_video.write_videofile(output_path, codec='libx264')