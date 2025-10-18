# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)
- 8GB+ RAM
- 10GB+ free disk space

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-ai-toolkit.git
cd face-ai-toolkit
```

### 2. Create Virtual Environment

```bash
python -m venv face_ai_env
source face_ai_env/bin/activate  # On Windows: face_ai_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Models

Create a `models` directory and download the following models:

```bash
mkdir models
cd models
```

Download these models:
- **InsightFace Model**: `inswapper_128.onnx`
- **GFPGAN Model**: `GFPGANv1.4.pth`
- **Real-ESRGAN Model**: `RealESRGAN_x4plus.pth`

### 5. Install the Package

```bash
pip install -e .
```

## GPU Setup (Recommended)

### CUDA Installation

1. Install CUDA Toolkit 11.8 or later
2. Install cuDNN
3. Verify installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Alternative: CPU-only Installation

If you don't have a GPU, install CPU-only versions:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime  # Instead of onnxruntime-gpu
```

## Verification

Run the basic example to verify installation:

```bash
python examples/basic_usage.py
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model not found**: Ensure models are downloaded to `models/` directory
3. **Import errors**: Check all dependencies are installed

### Performance Tips

- Use GPU for better performance
- Ensure sufficient RAM (8GB+)
- Close other applications during processing
- Use SSD storage for faster I/O

## Docker Installation (Alternative)

```bash
docker build -t face-ai-toolkit .
docker run --gpus all -v $(pwd):/workspace face-ai-toolkit
```

## Support

For installation issues, please check:
1. System requirements
2. CUDA compatibility
3. Python version compatibility
4. Available disk space