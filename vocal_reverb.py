import torch
import numpy as np
from scipy.signal import convolve

class VocalReverbNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "reverb_type": (["hall", "plate", "room", "church"], {"default": "hall"}),
                "size": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1}),
                "wet_dry": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("reverb_audio",)
    FUNCTION = "apply_reverb"
    CATEGORY = "audio/vocal_processing"

    def apply_reverb(self, audio, reverb_type, size, wet_dry):
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
        
        # Generate appropriate impulse response
        ir = self.generate_impulse_response(reverb_type, size, sample_rate)
        
        # Process each channel
        processed = np.zeros_like(audio_np)
        for c in range(audio_np.shape[0]):
            processed[c] = self.process_channel(
                audio_np[c], ir, wet_dry
            )
            
        # Convert back to tensor
        processed_tensor = torch.from_numpy(processed).unsqueeze(0)
        return ({"samples": processed_tensor},)
    
    def generate_impulse_response(self, reverb_type, size, sample_rate):
        """Generate impulse response for different reverb types"""
        duration = int(2.0 * size * sample_rate)  # 2 seconds max
        
        if reverb_type == "hall":
            return self.create_hall_ir(duration, sample_rate, size)
        elif reverb_type == "plate":
            return self.create_plate_ir(duration, sample_rate, size)
        elif reverb_type == "room":
            return self.create_room_ir(duration, sample_rate, size)
        else:  # church
            return self.create_church_ir(duration, sample_rate, size)
    
    def create_hall_ir(self, length, sample_rate, size):
        """Create concert hall impulse response"""
        t = np.arange(length) / sample_rate
        decay = np.exp(-t / (0.8 * size))
        early_reflections = np.zeros(length)
        
        # Add early reflections
        reflection_times = [0.005, 0.012, 0.020, 0.028, 0.035]
        for rt in reflection_times:
            idx = int(rt * sample_rate)
            if idx < length:
                early_reflections[idx] = 0.6 * np.random.randn()
        
        # Late reverb (diffuse)
        late_reverb = 0.4 * np.random.randn(length) * decay
        
        return early_reflections + late_reverb
    
    def create_plate_ir(self, length, sample_rate, size):
        """Create plate reverb impulse response"""
        t = np.arange(length) / sample_rate
        decay = np.exp(-t / (0.7 * size))
        return decay * np.random.randn(length)
    
    def process_channel(self, audio, ir, wet_dry):
        """Apply convolution reverb"""
        wet = convolve(audio, ir, mode='same')
        wet = wet / np.max(np.abs(wet))  # Normalize
        
        # Mix with dry signal
        return (1 - wet_dry) * audio + wet_dry * wet