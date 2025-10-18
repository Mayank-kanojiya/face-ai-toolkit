# API Documentation

## FaceAI Class

Main interface for all face manipulation operations.

### Constructor

```python
FaceAI(device='cuda')
```

**Parameters:**
- `device` (str): Computing device ('cuda' or 'cpu')

### Methods

#### face_swap(source_path, target_path, output_path)

Replace face in target image with source face.

**Parameters:**
- `source_path` (str): Path to source face image
- `target_path` (str): Path to target image
- `output_path` (str): Path for output image

**Returns:** str - Path to output file

#### deep_swap(source_path, target_path, output_path)

Advanced face swapping with neural network blending.

**Parameters:**
- `source_path` (str): Path to source face image
- `target_path` (str): Path to target image  
- `output_path` (str): Path for output image

**Returns:** str - Path to output file

#### enhance_face(input_path, output_path, scale=2)

Enhance facial features and quality.

**Parameters:**
- `input_path` (str): Path to input image
- `output_path` (str): Path for output image
- `scale` (int): Enhancement scale factor

**Returns:** str - Path to output file

#### enhance_frame(input_path, output_path, scale=4)

Upscale and enhance image or video frames.

**Parameters:**
- `input_path` (str): Path to input file
- `output_path` (str): Path for output file
- `scale` (int): Upscaling factor

**Returns:** str - Path to output file

#### lip_sync(video_path, audio_path, output_path)

Synchronize lip movements with audio.

**Parameters:**
- `video_path` (str): Path to input video
- `audio_path` (str): Path to audio file
- `output_path` (str): Path for output video

**Returns:** str - Path to output file

#### mouth_sync(video_path, audio_path, output_path)

Advanced mouth movement synchronization.

**Parameters:**
- `video_path` (str): Path to input video
- `audio_path` (str): Path to audio file
- `output_path` (str): Path for output video

**Returns:** str - Path to output file

#### age_transform(input_path, output_path, target_age)

Transform face to target age.

**Parameters:**
- `input_path` (str): Path to input image
- `output_path` (str): Path for output image
- `target_age` (int): Target age (5-80)

**Returns:** str - Path to output file

## Individual Classes

### FaceSwapper

Basic face swapping functionality.

```python
swapper = FaceSwapper(device='cuda')
swapper.swap(source_path, target_path, output_path)
swapper.swap_video(source_path, target_video_path, output_path)
```

### DeepSwapper

Advanced face swapping with neural networks.

```python
deep_swapper = DeepSwapper(device='cuda')
deep_swapper.swap(source_path, target_path, output_path)
```

### FaceEnhancer

Face restoration and enhancement.

```python
enhancer = FaceEnhancer(device='cuda')
enhancer.enhance(input_path, output_path, scale=2)
enhancer.batch_enhance(input_dir, output_dir)
```

### FrameEnhancer

Image and video upscaling.

```python
frame_enhancer = FrameEnhancer(device='cuda')
frame_enhancer.enhance_image(input_path, output_path, scale=4)
frame_enhancer.enhance_video(input_path, output_path, scale=4)
```

### LipSyncer

Lip synchronization with audio.

```python
lip_syncer = LipSyncer(device='cuda')
lip_syncer.sync(video_path, audio_path, output_path)
```

### MouthSyncer

Advanced mouth synchronization.

```python
mouth_syncer = MouthSyncer(device='cuda')
mouth_syncer.sync(video_path, audio_path, output_path)
```

### AgeSelector

Age transformation functionality.

```python
age_selector = AgeSelector(device='cuda')
age_selector.transform(input_path, output_path, target_age)
```

## Error Handling

All methods may raise the following exceptions:

- `ValueError`: Invalid input parameters or no faces detected
- `FileNotFoundError`: Input files not found
- `RuntimeError`: Processing errors or insufficient resources
- `torch.cuda.OutOfMemoryError`: GPU memory insufficient

## Example Usage

```python
from face_ai_toolkit import FaceAI

# Initialize
face_ai = FaceAI(device='cuda')

try:
    # Face swap
    result = face_ai.face_swap('source.jpg', 'target.jpg', 'output.jpg')
    print(f"Face swap completed: {result}")
    
    # Age transformation
    result = face_ai.age_transform('person.jpg', 'aged.jpg', target_age=65)
    print(f"Age transformation completed: {result}")
    
except ValueError as e:
    print(f"Invalid input: {e}")
except Exception as e:
    print(f"Processing error: {e}")
```