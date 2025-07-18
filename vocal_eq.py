import torch
import numpy as np
from scipy.signal import butter, sosfilt

class VocalEQNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "low_cut": ("FLOAT", {"default": 80.0, "min": 20.0, "max": 200.0, "step": 5.0}),
                "presence_boost": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                "presence_freq": ("FLOAT", {"default": 3000.0, "min": 1000.0, "max": 6000.0, "step": 100.0}),
                "air_boost": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                "air_freq": ("FLOAT", {"default": 12000.0, "min": 8000.0, "max": 20000.0, "step": 100.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("eq_audio",)
    FUNCTION = "apply_eq"
    CATEGORY = "audio/vocal_processing"

    def apply_eq(self, audio, low_cut, presence_boost, presence_freq, air_boost, air_freq):
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
        
        # Process each channel
        processed = np.zeros_like(audio_np)
        for c in range(audio_np.shape[0]):
            processed[c] = self.process_channel(
                audio_np[c], sample_rate, 
                low_cut, presence_boost, presence_freq, air_boost, air_freq
            )
            
        # Convert back to tensor
        processed_tensor = torch.from_numpy(processed).unsqueeze(0)
        return ({"samples": processed_tensor},)
    
    def process_channel(self, audio, sample_rate, low_cut, presence_boost, presence_freq, air_boost, air_freq):
        # Apply high-pass filter for low cut
        sos_high = butter(2, low_cut, 'highpass', fs=sample_rate, output='sos')
        audio = sosfilt(sos_high, audio)
        
        # Apply presence boost (bell curve)
        center = presence_freq
        q = 1.0
        gain = presence_boost
        sos_presence = self.create_peaking_eq(sample_rate, center, q, gain)
        audio = sosfilt(sos_presence, audio)
        
        # Apply air boost (high shelf)
        sos_air = self.create_high_shelf(sample_rate, air_freq, 0.7, air_boost)
        audio = sosfilt(sos_air, audio)
        
        return audio
    
    def create_peaking_eq(self, fs, f0, q, gain):
        """Create peaking EQ filter coefficients"""
        A = 10**(gain/40)
        w0 = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2 * q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        return np.array([[b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]])
    
    def create_high_shelf(self, fs, f0, s, gain):
        """Create high shelf filter coefficients"""
        A = 10**(gain/40)
        w0 = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/s - 1) + 2)
        
        b0 = A * ((A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
        b1 = -2*A * ((A-1) + (A+1)*np.cos(w0))
        b2 = A * ((A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
        a0 = (A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha
        a1 = 2 * ((A-1) - (A+1)*np.cos(w0))
        a2 = (A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha
        
        return np.array([[b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]])