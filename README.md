# Face AI Toolkit

A comprehensive AI-powered face manipulation toolkit built with FaceFusion's proven architecture. Features face swap, deep swap, face enhancement, frame enhancement, lip sync, mouth sync, and age transformation using state-of-the-art ONNX models.

## 🎯 Features

- **🔄 Face Swap**: Multiple models (InsightFace, SimSwap, Ghost) for face replacement
- **🎭 Deep Swap**: DeepFaceLive models for celebrity face swapping
- **✨ Face Enhancement**: GFPGAN, CodeFormer, GPEN for face restoration
- **🔍 Frame Enhancement**: Real-ESRGAN, UltraSharp for video/image upscaling
- **👄 Lip Sync**: Wav2Lip and EDTalk for audio-driven lip synchronization
- **👴 Age Transformation**: StyleGANEx for realistic age modification
- **🚀 ONNX Optimized**: Fast inference with GPU acceleration

## 🛠 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/face-ai-toolkit.git
cd face-ai-toolkit
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Models
```bash
# Download essential models (recommended)
python download_models.py --essential

# Or download all models (requires ~3GB+ storage)
python download_models.py --all

# List available models
python download_models.py --list
```

## 🚀 Quick Start

```python
from face_ai_toolkit import FaceAI

# Initialize with GPU acceleration
face_ai = FaceAI(device='cuda')

# Face swap with multiple model options
face_ai.face_swap("source.jpg", "target.jpg", "output.jpg")

# Deep face swap with celebrity models
face_ai.deep_swap("target.jpg", "celebrity_swap.jpg", "output.jpg")

# Face enhancement with GFPGAN
face_ai.enhance_face("low_quality.jpg", "enhanced.jpg")

# Frame enhancement with Real-ESRGAN
face_ai.enhance_frame("video.mp4", "upscaled_video.mp4", scale=4)

# Lip sync with Wav2Lip
face_ai.lip_sync("person.mp4", "speech.wav", "synced.mp4")

# Age transformation with StyleGANEx
face_ai.age_transform("person.jpg", "aged.jpg", target_age=65)
```

## 🎨 Advanced Usage

### Model Selection
```python
from face_ai_toolkit import FaceSwapper, FaceEnhancer

# Use specific face swap model
swapper = FaceSwapper(device='cuda')
swapper.set_model('simswap_256')  # or 'ghost_1_256', 'inswapper_128'

# Use specific enhancement model
enhancer = FaceEnhancer(device='cuda')
enhancer.set_model('codeformer')  # or 'gfpgan_1.4', 'gpen_bfr_512'
```

### Batch Processing
```python
# Batch enhance multiple images
enhancer.batch_enhance('input_folder/', 'output_folder/')
```

## 📋 Available Models

### Face Swappers
- **inswapper_128.onnx** (256MB) - Fast, general purpose
- **simswap_256.onnx** (285MB) - High quality results
- **ghost_1_256.onnx** (268MB) - Balanced quality/speed

### Face Enhancers
- **gfpgan_1.4.onnx** (348MB) - Best overall quality
- **codeformer.onnx** (376MB) - Robust to artifacts
- **gfpgan_1.3.onnx** (348MB) - Alternative version

### Frame Enhancers
- **real_esrgan_x4.onnx** (67MB) - 4x upscaling
- **real_esrgan_x2.onnx** (67MB) - 2x upscaling
- **span_kendata_x4.onnx** (67MB) - Anime/cartoon optimized

### Lip Syncers
- **wav2lip_gan_96.onnx** (44MB) - High quality lip sync
- **wav2lip_96.onnx** (44MB) - Standard version

### Age Modifiers
- **styleganex_age.onnx** (1.7GB) - Realistic age transformation

## 💻 System Requirements

- **Python**: 3.8+
- **GPU**: CUDA-compatible (recommended)
- **RAM**: 8GB+ (16GB+ for large models)
- **Storage**: 3GB+ for all models
- **OS**: Windows, Linux, macOS

## 📚 Documentation

- [Installation Guide](INSTALL.md)
- [API Reference](API.md)
- [Examples](examples/README.md)

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with inspiration from [FaceFusion](https://github.com/facefusion/facefusion) - an excellent face swapping framework.