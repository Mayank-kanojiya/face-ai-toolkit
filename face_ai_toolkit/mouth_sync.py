import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean
import librosa

class MouthSyncer:
    def __init__(self, device='cuda'):
        self.device = device
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Detailed mouth landmark indices
        self.outer_lip_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
    
    def sync(self, video_path, audio_path, output_path):
        # Extract detailed audio features
        audio_data = self.extract_detailed_audio_features(audio_path)
        
        # Process video with advanced mouth tracking
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = 'temp_mouth_synced.mp4'
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get phoneme prediction for current frame
                phoneme_data = self.get_phoneme_at_frame(audio_data, frame_idx, fps)
                
                # Apply advanced mouth morphing
                modified_frame = self.morph_mouth_advanced(frame, face_landmarks, phoneme_data)
                out.write(modified_frame)
            else:
                out.write(frame)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Combine with original audio
        from moviepy.editor import VideoFileClip, AudioFileClip
        video = VideoFileClip(temp_video)
        audio = AudioFileClip(audio_path)
        final_video = video.set_audio(audio)
        final_video.write_videofile(output_path, codec='libx264')
        
        return output_path
    
    def extract_detailed_audio_features(self, audio_path):
        y, sr = librosa.load(audio_path)
        
        # Extract multiple audio features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        return {
            'mfccs': mfccs,
            'spectral_centroids': spectral_centroids,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'rms': rms,
            'sr': sr
        }
    
    def get_phoneme_at_frame(self, audio_data, frame_idx, fps):
        time_per_frame = 1.0 / fps
        audio_idx = int(frame_idx * time_per_frame * audio_data['sr'] / 512)
        
        if audio_idx < len(audio_data['rms']):
            # Classify mouth shape based on audio features
            rms = audio_data['rms'][audio_idx]
            zcr = audio_data['zero_crossing_rate'][audio_idx] if audio_idx < len(audio_data['zero_crossing_rate']) else 0
            
            # Simple phoneme classification
            if rms > 0.02:
                if zcr > 0.1:
                    return 'open'  # Vowels
                else:
                    return 'closed'  # Consonants
            else:
                return 'neutral'
        
        return 'neutral'
    
    def morph_mouth_advanced(self, frame, face_landmarks, phoneme_type):
        h, w = frame.shape[:2]
        
        # Get mouth landmarks
        outer_lip_points = self.get_landmark_points(face_landmarks, self.outer_lip_indices, w, h)
        inner_lip_points = self.get_landmark_points(face_landmarks, self.inner_lip_indices, w, h)
        
        # Calculate mouth metrics
        mouth_width = euclidean(outer_lip_points[0], outer_lip_points[6])
        mouth_height = euclidean(outer_lip_points[3], outer_lip_points[9])
        
        # Apply morphing based on phoneme type
        if phoneme_type == 'open':
            # Increase mouth opening
            scale_factor = 1.2
        elif phoneme_type == 'closed':
            # Decrease mouth opening
            scale_factor = 0.8
        else:
            # Neutral position
            scale_factor = 1.0
        
        # Apply subtle mouth shape modification
        modified_frame = self.apply_mouth_warp(frame, outer_lip_points, scale_factor)
        
        return modified_frame
    
    def get_landmark_points(self, face_landmarks, indices, w, h):
        points = []
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        return np.array(points)
    
    def apply_mouth_warp(self, frame, mouth_points, scale_factor):
        # Simple mouth region warping
        mouth_center = np.mean(mouth_points, axis=0).astype(int)
        
        # Create a subtle warp effect
        rows, cols = frame.shape[:2]
        
        # Define source and destination points for warping
        src_points = mouth_points.astype(np.float32)
        dst_points = mouth_points.copy().astype(np.float32)
        
        # Scale mouth points
        for i, point in enumerate(dst_points):
            direction = point - mouth_center
            dst_points[i] = mouth_center + direction * scale_factor
        
        # Apply perspective transform to mouth region only
        if len(src_points) >= 4:
            # Use only 4 points for perspective transform
            src_quad = src_points[:4]
            dst_quad = dst_points[:4]
            
            M = cv2.getPerspectiveTransform(src_quad, dst_quad)
            
            # Create mask for mouth region
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [mouth_points.astype(int)], 255)
            
            # Apply transform only to mouth region
            warped = cv2.warpPerspective(frame, M, (cols, rows))
            result = np.where(mask[..., None], warped, frame)
            
            return result
        
        return frame