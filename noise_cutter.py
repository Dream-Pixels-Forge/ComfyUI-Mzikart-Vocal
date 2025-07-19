import os
import torch
import numpy as np
from datetime import datetime
import folder_paths

# Try to import noisereduce
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("Warning: noisereduce package not found. Noise reduction will be disabled.")
    print("To enable noise reduction, run the install_noisereduce.bat script or use:")
    print("pip install noisereduce")

class MzikartNoiseCutter:
    """
    Remove background noise from audio files using spectral gating with presets
    """
    
    PRESETS = {
        "Gentle Vocal Cleanup": {"threshold": 0.2, "duration": 0.5, "aggressiveness": 8},
        "Vocal Booth Quality": {"threshold": 0.3, "duration": 0.6, "aggressiveness": 12},
        "Instrumental Cleanup": {"threshold": 0.15, "duration": 0.5, "aggressiveness": 6},
        "Podcast/Interview": {"threshold": 0.25, "duration": 1.0, "aggressiveness": 10},
        "Field Recording": {"threshold": 0.4, "duration": 1.5, "aggressiveness": 15},
        "Aggressive Noise Removal": {"threshold": 0.8, "duration": 2.0, "aggressiveness": 20},
        "Custom": {"threshold": 0.1, "duration": 0.5, "aggressiveness": 10}
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": ("STRING", {"default": ""}),
                "preset": (list(cls.PRESETS.keys()), {"default": "Gentle Vocal Cleanup"}),
                "noise_threshold": ("FLOAT", {
                    "default": cls.PRESETS["Gentle Vocal Cleanup"]["threshold"], 
                    "min": 0.01, 
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "noise_duration": ("FLOAT", {
                    "default": cls.PRESETS["Gentle Vocal Cleanup"]["duration"], 
                    "min": 0.1, 
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "aggressiveness": ("FLOAT", {
                    "default": cls.PRESETS["Gentle Vocal Cleanup"]["aggressiveness"], 
                    "min": 1, 
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
                "output_format": (["wav", "mp3"], {"default": "wav"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("denoised_audio_path",)
    FUNCTION = "denoise_audio"
    CATEGORY = "Mzikart/Audio"

    def denoise_audio(self, audio_file, preset, noise_threshold, noise_duration, aggressiveness, output_format):
        # Check if noisereduce is available
        if not HAS_NOISEREDUCE:
            raise ImportError("The noisereduce package is required for noise reduction. Please install it using: pip install noisereduce")
            
        # Apply preset if not custom
        if preset != "Custom":
            settings = self.PRESETS[preset]
            noise_threshold = settings["threshold"]
            noise_duration = settings["duration"]
            aggressiveness = settings["aggressiveness"]
            print(f"Applying preset: {preset} - Threshold: {noise_threshold}, Duration: {noise_duration}s, Aggressiveness: {aggressiveness}")
        
        # Validate inputs
        if not audio_file:
            raise ValueError("Audio file path must be provided")
        
        try:
            import torchaudio
        except ImportError:
            raise ImportError("The torchaudio package is required. Please install it using: pip install torchaudio")
            
        # Get absolute path
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, audio_file) if not os.path.isabs(audio_file) else audio_file
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Convert to numpy array
        audio_np = waveform.numpy().squeeze()
        
        # Extract noise profile (first portion of audio)
        noise_samples = int(sample_rate * noise_duration)
        noise_clip = audio_np[:min(noise_samples, len(audio_np))]
        
        # Apply noise reduction
        denoised_audio = nr.reduce_noise(
            y=audio_np,
            sr=sample_rate,
            y_noise=noise_clip,
            prop_decrease=noise_threshold,
            n_std_thresh=aggressiveness,
            stationary=True
        )
        
        # Convert back to tensor
        denoised_tensor = torch.tensor(denoised_audio).float().unsqueeze(0)
        
        # Create output filename
        output_dir = folder_paths.get_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"denoised_audio_{timestamp}.{output_format}"
        output_path = os.path.join(output_dir, filename)
        
        # Save result
        torchaudio.save(output_path, denoised_tensor, sample_rate)
        
        return (output_path,)