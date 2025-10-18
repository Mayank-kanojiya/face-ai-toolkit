import cv2
import numpy as np
from .face_swap import FaceSwapper
from .deep_swap import DeepSwapper
from .face_enhance import FaceEnhancer
from .frame_enhance import FrameEnhancer
from .lip_sync import LipSyncer
from .mouth_sync import MouthSyncer
from .age_selector import AgeSelector

class FaceAI:
    def __init__(self, device='cuda'):
        self.device = device
        self.face_swapper = FaceSwapper(device)
        self.deep_swapper = DeepSwapper(device)
        self.face_enhancer = FaceEnhancer(device)
        self.frame_enhancer = FrameEnhancer(device)
        self.lip_syncer = LipSyncer(device)
        self.mouth_syncer = MouthSyncer(device)
        self.age_selector = AgeSelector(device)
    
    def face_swap(self, source_path, target_path, output_path):
        return self.face_swapper.swap(source_path, target_path, output_path)
    
    def deep_swap(self, source_path, target_path, output_path):
        return self.deep_swapper.swap(source_path, target_path, output_path)
    
    def enhance_face(self, input_path, output_path, scale=2):
        return self.face_enhancer.enhance(input_path, output_path, scale)
    
    def enhance_frame(self, input_path, output_path, scale=4):
        return self.frame_enhancer.enhance(input_path, output_path, scale)
    
    def lip_sync(self, video_path, audio_path, output_path):
        return self.lip_syncer.sync(video_path, audio_path, output_path)
    
    def mouth_sync(self, video_path, audio_path, output_path):
        return self.mouth_syncer.sync(video_path, audio_path, output_path)
    
    def age_transform(self, input_path, output_path, target_age):
        return self.age_selector.transform(input_path, output_path, target_age)