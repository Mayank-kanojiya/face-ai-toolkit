#!/usr/bin/env python3
"""
Basic usage examples for Face AI Toolkit
"""

from face_ai_toolkit import FaceAI
import os

def main():
    # Initialize the Face AI toolkit
    face_ai = FaceAI(device='cuda')  # Use 'cpu' if no GPU available
    
    # Example 1: Face Swap
    print("Running Face Swap example...")
    try:
        face_ai.face_swap(
            source_path="examples/images/source_face.jpg",
            target_path="examples/images/target_person.jpg", 
            output_path="examples/outputs/face_swapped.jpg"
        )
        print("✓ Face swap completed")
    except Exception as e:
        print(f"✗ Face swap failed: {e}")
    
    # Example 2: Deep Swap
    print("Running Deep Swap example...")
    try:
        face_ai.deep_swap(
            source_path="examples/images/source_face.jpg",
            target_path="examples/images/target_person.jpg",
            output_path="examples/outputs/deep_swapped.jpg"
        )
        print("✓ Deep swap completed")
    except Exception as e:
        print(f"✗ Deep swap failed: {e}")
    
    # Example 3: Face Enhancement
    print("Running Face Enhancement example...")
    try:
        face_ai.enhance_face(
            input_path="examples/images/low_quality_face.jpg",
            output_path="examples/outputs/enhanced_face.jpg",
            scale=2
        )
        print("✓ Face enhancement completed")
    except Exception as e:
        print(f"✗ Face enhancement failed: {e}")
    
    # Example 4: Frame Enhancement
    print("Running Frame Enhancement example...")
    try:
        face_ai.enhance_frame(
            input_path="examples/videos/low_res_video.mp4",
            output_path="examples/outputs/enhanced_video.mp4",
            scale=4
        )
        print("✓ Frame enhancement completed")
    except Exception as e:
        print(f"✗ Frame enhancement failed: {e}")
    
    # Example 5: Lip Sync
    print("Running Lip Sync example...")
    try:
        face_ai.lip_sync(
            video_path="examples/videos/person_talking.mp4",
            audio_path="examples/audio/new_speech.wav",
            output_path="examples/outputs/lip_synced.mp4"
        )
        print("✓ Lip sync completed")
    except Exception as e:
        print(f"✗ Lip sync failed: {e}")
    
    # Example 6: Mouth Sync
    print("Running Mouth Sync example...")
    try:
        face_ai.mouth_sync(
            video_path="examples/videos/person_talking.mp4",
            audio_path="examples/audio/new_speech.wav",
            output_path="examples/outputs/mouth_synced.mp4"
        )
        print("✓ Mouth sync completed")
    except Exception as e:
        print(f"✗ Mouth sync failed: {e}")
    
    # Example 7: Age Transformation
    print("Running Age Transformation example...")
    try:
        # Make person look 25 years old
        face_ai.age_transform(
            input_path="examples/images/person.jpg",
            output_path="examples/outputs/aged_25.jpg",
            target_age=25
        )
        
        # Make person look 65 years old
        face_ai.age_transform(
            input_path="examples/images/person.jpg",
            output_path="examples/outputs/aged_65.jpg",
            target_age=65
        )
        print("✓ Age transformation completed")
    except Exception as e:
        print(f"✗ Age transformation failed: {e}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("examples/outputs", exist_ok=True)
    main()