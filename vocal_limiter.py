import torch
import numpy as np

class VocalLimiterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold": ("FLOAT", {"default": -0.3, "min": -12.0, "max": 0.0, "step": 0.1}),
                "release": ("FLOAT", {"default": 30.0, "min": 5.0, "max": 500.0, "step": 5.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("limited_audio",)
    FUNCTION = "limit"
    CATEGORY = "audio/vocal_processing"

    def limit(self, audio, threshold, release):
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
                threshold, release
            )
            
        # Convert back to tensor
        processed_tensor = torch.from_numpy(processed).unsqueeze(0)
        return ({"samples": processed_tensor},)
    
    def process_channel(self, audio, sample_rate, threshold, release):
        threshold_linear = 10 ** (threshold / 20)
        release_samples = int((release / 1000) * sample_rate)
        
        gain = 1.0
        envelope = 0.0
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Detect peaks
            peak = np.abs(audio[i])
            
            # Calculate gain reduction needed
            if peak > threshold_linear:
                reduction_needed = threshold_linear / peak
            else:
                reduction_needed = 1.0
                
            # Smooth gain changes
            if reduction_needed < gain:
                # Attack immediately
                gain = reduction_needed
            else:
                # Release phase
                gain += (1.0 - gain) / release_samples
                
            # Apply gain reduction
            output[i] = audio[i] * gain
                
        return output