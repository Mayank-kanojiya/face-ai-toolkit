# Quick fix for Colab import issues
# Run this cell first in Colab

# Install missing dependencies
!pip install mediapipe
!pip install --upgrade moviepy

# Fix mouth_sync import issue
import os
mouth_sync_path = '/content/face-ai-toolkit/face_ai_toolkit/mouth_sync.py'

if os.path.exists(mouth_sync_path):
    # Read and fix the file
    with open(mouth_sync_path, 'r') as f:
        content = f.read()
    
    # Replace mediapipe import with fallback
    fixed_content = content.replace(
        'import mediapipe as mp',
        '''try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe not available, using fallback")
    mp = None'''
    )
    
    # Write fixed content
    with open(mouth_sync_path, 'w') as f:
        f.write(fixed_content)
    
    print("✅ Fixed mouth_sync.py import issue")

# Now import the toolkit
import sys
sys.path.append('/content/face-ai-toolkit')

from face_ai_toolkit import FaceAI
print("✅ Face AI Toolkit imported successfully!")