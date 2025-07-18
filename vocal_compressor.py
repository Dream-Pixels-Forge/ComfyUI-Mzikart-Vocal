import torch
import numpy as np
from scipy.signal import lfilter

class VocalCompressorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold": ("FLOAT", {"default": -25.0, "min": -60.0, "max": 0.0, "step": 1.0}),
                "ratio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "attack": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "release": ("FLOAT", {"default": 150.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
                "makeup_gain": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 12.0, "step": 0.5}),
                "knee": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                "genre": (["general", "rap", "rnb", "gospel", "pop"], {"default": "general"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("compressed_audio",)
    FUNCTION = "compress"
    CATEGORY = "audio/vocal_processing"

    def compress(self, audio, threshold, ratio, attack, release, makeup_gain, knee, genre):
        # Get audio tensor
        if isinstance(audio, dict) and 'samples' in audio:
            audio_tensor = audio['samples']
        else:
            audio_tensor = audio
        
        # Apply genre-specific presets
        if genre == "rap":
            threshold = max(threshold, -20.0)
            ratio = min(ratio + 1.0, 8.0)
            attack = max(attack, 5.0)
        elif genre == "rnb":
            threshold = min(threshold, -22.0)
            ratio = max(ratio - 0.5, 2.0)
            release = min(release + 50.0, 300.0)
        elif genre == "gospel":
            knee = min(knee + 2.0, 8.0)
            makeup_gain = min(makeup_gain + 1.0, 6.0)
        
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
                threshold, ratio, attack, release, 
                makeup_gain, knee
            )
            
        # Convert back to tensor
        processed_tensor = torch.from_numpy(processed).unsqueeze(0)
        return ({"samples": processed_tensor},)

    def process_channel(self, audio, sample_rate, threshold, ratio, attack, release, makeup_gain, knee):
        # Convert parameters
        attack_samples = int((attack / 1000) * sample_rate)
        release_samples = int((release / 1000) * sample_rate)
        threshold_linear = 10 ** (threshold / 20)
        makeup_gain_linear = 10 ** (makeup_gain / 20)
        
        # Initialize variables
        gain_reduction = 0.0
        envelope = 0.0
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Calculate signal level (RMS with 10ms window)
            start_idx = max(0, i - int(0.01 * sample_rate))
            rms = np.sqrt(np.mean(audio[start_idx:i+1]**2))
            db = 20 * np.log10(max(rms, 1e-7))
            
            # Calculate overshoot
            overshoot = db - threshold
            if overshoot < 0:
                overshoot = 0
                
            # Soft knee
            if knee > 0 and overshoot > 0 and overshoot < knee:
                overshoot = overshoot**2 / (2 * knee)
            
            # Calculate desired gain reduction
            desired_reduction = overshoot * (1 - 1/ratio)
            
            # Attack/release envelope
            if desired_reduction > gain_reduction:
                # Attack phase
                gain_reduction += (desired_reduction - gain_reduction) / attack_samples
            else:
                # Release phase
                gain_reduction -= (gain_reduction - desired_reduction) / release_samples
                
            # Convert gain reduction to linear scale
            gain_linear = makeup_gain_linear * (10 ** (-gain_reduction / 20))
            
            # Apply gain reduction
            output[i] = audio[i] * gain_linear
                
        return output