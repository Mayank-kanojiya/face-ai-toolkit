#!/usr/bin/env python3
"""
Model downloader for Face AI Toolkit
Downloads required models for all features
"""

import os
import requests
from tqdm import tqdm
from pathlib import Path

class ModelDownloader:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Working model URLs
        self.models = {
            # Face Swappers
            'inswapper_128_fp16.onnx': {
                'url': 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx',
                'size': '256MB'
            },
            'inswapper_128.onnx': {
                'url': 'https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx',
                'size': '256MB'
            },
            
            # Face Enhancers
            'GFPGANv1.4.pth': {
                'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
                'size': '348MB'
            },
            'GFPGANv1.3.pth': {
                'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                'size': '348MB'
            },
            
            # Frame Enhancers
            'RealESRGAN_x4plus.pth': {
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                'size': '67MB'
            },
            'RealESRGAN_x2plus.pth': {
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                'size': '67MB'
            },
            
            # Lip Syncers
            'wav2lip_gan.pth': {
                'url': 'https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth',
                'size': '44MB'
            },
            'wav2lip.pth': {
                'url': 'https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth',
                'size': '44MB'
            }
        }
    
    def download_file(self, url: str, filepath: Path, description: str = ""):
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Downloaded: {filepath.name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {filepath.name}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_essential_models(self):
        """Download essential models for basic functionality"""
        essential_models = [
            'inswapper_128_fp16.onnx',
            'GFPGANv1.4.pth', 
            'RealESRGAN_x4plus.pth',
            'wav2lip_gan.pth'
        ]
        
        print("Downloading essential models...")
        for model_name in essential_models:
            if model_name in self.models:
                self._download_model(model_name)
    
    def download_all_models(self):
        """Download all available models"""
        print("Downloading all models...")
        for model_name in self.models:
            self._download_model(model_name)
    
    def _download_model(self, model_name: str):
        """Download a single model"""
        model_info = self.models[model_name]
        filepath = self.models_dir / model_name
        
        if filepath.exists():
            print(f"⚠ Model already exists: {model_name}")
            return
        
        print(f"Downloading {model_name} ({model_info['size']})...")
        self.download_file(
            model_info['url'], 
            filepath, 
            f"Downloading {model_name}"
        )
    
    def list_models(self):
        """List all available models"""
        print("\n=== Available Models ===")
        
        print("\n🔄 Face Swappers:")
        for name in ['inswapper_128_fp16.onnx', 'inswapper_128.onnx']:
            if name in self.models:
                print(f"  - {name} ({self.models[name]['size']})")
        
        print("\n✨ Face Enhancers:")
        for name in ['GFPGANv1.4.pth', 'GFPGANv1.3.pth']:
            if name in self.models:
                print(f"  - {name} ({self.models[name]['size']})")
        
        print("\n🔍 Frame Enhancers:")
        for name in ['RealESRGAN_x4plus.pth', 'RealESRGAN_x2plus.pth']:
            if name in self.models:
                print(f"  - {name} ({self.models[name]['size']})")
        
        print("\n👄 Lip Syncers:")
        for name in ['wav2lip_gan.pth', 'wav2lip.pth']:
            if name in self.models:
                print(f"  - {name} ({self.models[name]['size']})")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Face AI Toolkit models")
    parser.add_argument('--essential', action='store_true', help='Download only essential models')
    parser.add_argument('--all', action='store_true', help='Download all models')
    parser.add_argument('--list', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.list:
        downloader.list_models()
    elif args.essential:
        downloader.download_essential_models()
    elif args.all:
        downloader.download_all_models()
    else:
        print("Face AI Toolkit Model Downloader")
        print("Usage:")
        print("  python download_models.py --essential  # Download essential models")
        print("  python download_models.py --all        # Download all models")
        print("  python download_models.py --list       # List available models")

if __name__ == "__main__":
    main()