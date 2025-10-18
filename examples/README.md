# Examples

This directory contains example scripts and sample data for the Face AI Toolkit.

## Directory Structure

```
examples/
├── basic_usage.py          # Basic usage examples
├── advanced_usage.py       # Advanced pipeline examples
├── images/                 # Sample input images
├── videos/                 # Sample input videos
├── audio/                  # Sample audio files
└── outputs/               # Generated output files
```

## Running Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

This script demonstrates:
- Face swapping
- Deep face swapping
- Face enhancement
- Frame enhancement
- Lip synchronization
- Mouth synchronization
- Age transformation

### Advanced Usage

```bash
python examples/advanced_usage.py
```

This script demonstrates:
- Advanced face swap pipelines
- Batch processing
- Video processing pipelines
- Age progression series
- Custom video face swapping

## Sample Data

To run the examples, you'll need to add sample files:

### Images (examples/images/)
- `source_face.jpg` - Source face for swapping
- `target_person.jpg` - Target person image
- `low_quality_face.jpg` - Low quality face for enhancement
- `person.jpg` - Person for age transformation

### Videos (examples/videos/)
- `low_res_video.mp4` - Low resolution video for enhancement
- `person_talking.mp4` - Video of person talking
- `target_video.mp4` - Target video for face swapping

### Audio (examples/audio/)
- `new_speech.wav` - Audio for lip sync
- `new_audio.wav` - Audio for mouth sync

## Expected Outputs

After running the examples, you'll find generated files in `examples/outputs/`:

- `face_swapped.jpg` - Basic face swap result
- `deep_swapped.jpg` - Deep swap result
- `enhanced_face.jpg` - Enhanced face
- `enhanced_video.mp4` - Enhanced video
- `lip_synced.mp4` - Lip synced video
- `mouth_synced.mp4` - Mouth synced video
- `aged_25.jpg` - Age transformed to 25
- `aged_65.jpg` - Age transformed to 65

## Notes

- Ensure you have sufficient GPU memory for video processing
- Processing times vary based on input size and hardware
- Some examples require specific model files in the `models/` directory