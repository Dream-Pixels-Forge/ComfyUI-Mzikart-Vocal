import torch
import numpy as np
from scipy.signal import butter, sosfilt

class VocalDeesserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sensitivity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "frequency": ("FLOAT", {"default": 6000.0, "min": 3000.0, "max": 8000.0, "step": 100.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("deessed_audio",)
    FUNCTION = "deess"
    CATEGORY = "audio/vocal_processing"

    def deess(self, audio, sensitivity, frequency):
        # Get audio tensor
        if isinstance(audio, dict) and 'samples' in audio:
            audio_tensor = audio['samples']
        else:
            audio_tensor = audio
            
        # Convert to numpy array
        if isinstance(audio_tensor, torch.Tensor) and audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)
        audio_np = audio_tensor.numpy()
        sample_rate = 48000
        
        # Create band-pass filter for sibilance range
        sos = butter(4, [frequency-1000, frequency+1000], 'bandpass', fs=sample_rate, output='sos')
        
        # Process each channel
        processed = np.zeros_like(audio_np)
        for c in range(audio_np.shape[0]):
            processed[c] = self.process_channel(
                audio_np[c], sample_rate, sos, sensitivity
            )
            
        # Convert back to tensor
        processed_tensor = torch.from_numpy(processed).unsqueeze(0)
        return ({"samples": processed_tensor},)
    
    def process_channel(self, audio, sample_rate, sos, sensitivity):
        # Extract sibilance frequencies
        sibilance = sosfilt(sos, audio)
        
        # Detect sibilance peaks
        rms = np.sqrt(np.convolve(sibilance**2, np.ones(50)/50, mode='same'))
        threshold = np.percentile(rms, 95) * sensitivity
        reduction = np.where(rms > threshold, threshold / (rms + 1e-7), 1.0)
        
        # Apply reduction only to sibilance range
        return audio * reduction