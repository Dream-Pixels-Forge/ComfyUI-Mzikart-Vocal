import os
import torch
import numpy as np
from datetime import datetime
import folder_paths
import torchaudio

# Try to import noisereduce
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("Warning: noisereduce package not found. Noise reduction will be disabled.")
    print("To enable noise reduction, run the install_noisereduce.bat script or use:")
    print("pip install noisereduce")

# Try to import Facebook Research denoiser
try:
    from denoiser import pretrained
    from denoiser.dsp import convert_audio
    HAS_FB_DENOISER = True
except ImportError:
    HAS_FB_DENOISER = False
    print("Warning: Facebook Research denoiser not found. Advanced denoising will be disabled.")
    print("To enable advanced denoising, install it using:")
    print("pip install -U git+https://github.com/facebookresearch/denoiser")

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
        denoiser_types = ["NoiseReduce (Spectral Gating)"]
        if HAS_FB_DENOISER:
            denoiser_types.extend(["Facebook Denoiser (DNS48)", "Facebook Denoiser (DNS64)", "Facebook Denoiser (Master64)"])
            
        return {
            "required": {
                "audio_input": (["STRING", "TENSOR"], {"default": ""}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1, "display": "number"}),
                "denoiser_type": (denoiser_types, {"default": denoiser_types[0]}),
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

    def denoise_audio(self, audio_input, sample_rate, denoiser_type, preset, noise_threshold, noise_duration, aggressiveness, output_format):
        # Apply preset if not custom
        if preset != "Custom":
            settings = self.PRESETS[preset]
            noise_threshold = settings["threshold"]
            noise_duration = settings["duration"]
            aggressiveness = settings["aggressiveness"]
            print(f"Applying preset: {preset} - Threshold: {noise_threshold}, Duration: {noise_duration}s, Aggressiveness: {aggressiveness}")
        
        # Handle tensor input
        if isinstance(audio_input, torch.Tensor):
            waveform = audio_input
            # Ensure proper shape: [channels, samples]
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension if mono
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Handle file path input (legacy support)
        else:
            if not audio_input:
                raise ValueError("Either audio_input tensor or file path must be provided")
            
            if not torchaudio:
                raise ImportError("The torchaudio package is required for file input. Please install it using: pip install torchaudio")
                
            # Get absolute path
            input_dir = folder_paths.get_input_directory()
            file_path = os.path.join(input_dir, audio_input) if not os.path.isabs(audio_input) else audio_input
            
            # Load audio file
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Convert to 16kHz for Facebook denoiser if needed
        if denoiser_type != "NoiseReduce (Spectral Gating)":
            if not HAS_FB_DENOISER:
                raise ImportError("The Facebook Research denoiser is not installed. Please install it using: pip install -U git+https://github.com/facebookresearch/denoiser")
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Initialize the appropriate Facebook denoiser model
            if denoiser_type == "Facebook Denoiser (DNS48)":
                model = pretrained.dns48().cuda() if torch.cuda.is_available() else pretrained.dns48()
            elif denoiser_type == "Facebook Denoiser (DNS64)":
                model = pretrained.dns64().cuda() if torch.cuda.is_available() else pretrained.dns64()
            else:  # Master64
                model = pretrained.master64().cuda() if torch.cuda.is_available() else pretrained.master64()
            
            # Move to device and set to eval mode
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            
            # Convert audio to the right format for the model
            with torch.no_grad():
                # Ensure proper shape and device
                waveform = waveform.to(device)
                # Apply denoising
                denoised_waveform = model(waveform[None])[0]
                
            # Convert back to CPU if needed
            denoised_tensor = denoised_waveform.cpu() if torch.cuda.is_available() else denoised_waveform
            
        else:  # Use noisereduce
            if not HAS_NOISEREDUCE:
                raise ImportError("The noisereduce package is required for spectral gating. Please install it using: pip install noisereduce")
            
            # Convert to numpy array for noisereduce
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