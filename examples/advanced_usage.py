#!/usr/bin/env python3
"""
Advanced usage examples for Face AI Toolkit
"""

from face_ai_toolkit import (
    FaceSwapper, DeepSwapper, FaceEnhancer, 
    FrameEnhancer, LipSyncer, MouthSyncer, AgeSelector
)
import os

def advanced_face_swap_pipeline():
    """Advanced face swap with enhancement pipeline"""
    print("Running advanced face swap pipeline...")
    
    # Initialize components
    face_swapper = FaceSwapper(device='cuda')
    face_enhancer = FaceEnhancer(device='cuda')
    
    # Step 1: Perform face swap
    swapped_path = "examples/outputs/temp_swapped.jpg"
    face_swapper.swap(
        source_path="examples/images/source_face.jpg",
        target_path="examples/images/target_person.jpg",
        output_path=swapped_path
    )
    
    # Step 2: Enhance the swapped result
    face_enhancer.enhance(
        input_path=swapped_path,
        output_path="examples/outputs/enhanced_swap.jpg",
        scale=2
    )
    
    print("✓ Advanced face swap pipeline completed")

def batch_processing_example():
    """Batch process multiple images"""
    print("Running batch processing example...")
    
    face_enhancer = FaceEnhancer(device='cuda')
    
    # Create input and output directories
    input_dir = "examples/images/batch_input"
    output_dir = "examples/outputs/batch_output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Batch enhance all images in directory
    face_enhancer.batch_enhance(input_dir, output_dir)
    
    print("✓ Batch processing completed")

def video_processing_pipeline():
    """Complete video processing pipeline"""
    print("Running video processing pipeline...")
    
    # Initialize components
    frame_enhancer = FrameEnhancer(device='cuda')
    lip_syncer = LipSyncer(device='cuda')
    
    # Step 1: Enhance video quality
    enhanced_video = "examples/outputs/temp_enhanced.mp4"
    frame_enhancer.enhance_video(
        input_path="examples/videos/input_video.mp4",
        output_path=enhanced_video,
        scale=2
    )
    
    # Step 2: Apply lip sync
    lip_syncer.sync(
        video_path=enhanced_video,
        audio_path="examples/audio/new_audio.wav",
        output_path="examples/outputs/final_video.mp4"
    )
    
    print("✓ Video processing pipeline completed")

def age_progression_series():
    """Create age progression series"""
    print("Creating age progression series...")
    
    age_selector = AgeSelector(device='cuda')
    input_image = "examples/images/person.jpg"
    
    # Create progression from 20 to 80 years old
    ages = [20, 30, 40, 50, 60, 70, 80]
    
    for age in ages:
        output_path = f"examples/outputs/age_{age}.jpg"
        age_selector.transform(
            input_path=input_image,
            output_path=output_path,
            target_age=age
        )
    
    print("✓ Age progression series completed")

def custom_face_swap_with_video():
    """Face swap in video with custom settings"""
    print("Running custom video face swap...")
    
    face_swapper = FaceSwapper(device='cuda')
    
    # Swap faces in video
    face_swapper.swap_video(
        source_path="examples/images/source_face.jpg",
        target_video_path="examples/videos/target_video.mp4",
        output_path="examples/outputs/swapped_video.mp4"
    )
    
    print("✓ Custom video face swap completed")

def main():
    """Run all advanced examples"""
    # Create necessary directories
    os.makedirs("examples/outputs", exist_ok=True)
    os.makedirs("examples/images", exist_ok=True)
    os.makedirs("examples/videos", exist_ok=True)
    os.makedirs("examples/audio", exist_ok=True)
    
    try:
        advanced_face_swap_pipeline()
    except Exception as e:
        print(f"✗ Advanced face swap pipeline failed: {e}")
    
    try:
        batch_processing_example()
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
    
    try:
        video_processing_pipeline()
    except Exception as e:
        print(f"✗ Video processing pipeline failed: {e}")
    
    try:
        age_progression_series()
    except Exception as e:
        print(f"✗ Age progression series failed: {e}")
    
    try:
        custom_face_swap_with_video()
    except Exception as e:
        print(f"✗ Custom video face swap failed: {e}")

if __name__ == "__main__":
    main()