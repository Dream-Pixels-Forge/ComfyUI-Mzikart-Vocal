import torch
import numpy as np
from .utils import extract_audio_tensor

class MasteringCombinerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "mix_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "normalize": (["True", "False"], {"default": "True"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("combined_audio",)
    FUNCTION = "combine_audio"
    CATEGORY = "Mzikart/Mastering"

    def combine_audio(self, audio1, audio2, mix_ratio, normalize):
        try:
            # Extract audio tensors using the helper function
            audio1_tensor, sample_rate1, original_format1 = extract_audio_tensor(audio1)
            audio2_tensor, sample_rate2, original_format2 = extract_audio_tensor(audio2)
            
            # Convert to numpy for processing
            if audio1_tensor.dim() == 3:
                audio1_tensor = audio1_tensor.squeeze(0)
            audio1_np = audio1_tensor.numpy()
            
            if audio2_tensor.dim() == 3:
                audio2_tensor = audio2_tensor.squeeze(0)
            audio2_np = audio2_tensor.numpy()
        
            # Check if shapes match, if not, make them match
            if audio1_np.shape != audio2_np.shape:
                # Get the number of channels (use the max)
                channels = max(audio1_np.shape[0], audio2_np.shape[0])
                
                # Get the length (use the min)
                length = min(audio1_np.shape[1], audio2_np.shape[1])
                
                # Resize arrays
                audio1_resized = np.zeros((channels, length))
                audio2_resized = np.zeros((channels, length))
                
                # Copy data
                for c in range(min(channels, audio1_np.shape[0])):
                    audio1_resized[c, :min(length, audio1_np.shape[1])] = audio1_np[c, :min(length, audio1_np.shape[1])]
                    
                for c in range(min(channels, audio2_np.shape[0])):
                    audio2_resized[c, :min(length, audio2_np.shape[1])] = audio2_np[c, :min(length, audio2_np.shape[1])]
                    
                audio1_np = audio1_resized
                audio2_np = audio2_resized
            
            # Mix the audio
            combined = (1 - mix_ratio) * audio1_np + mix_ratio * audio2_np
            
            # Normalize if requested
            if normalize == "True":
                max_val = np.max(np.abs(combined))
                if max_val > 0:
                    combined = combined / max_val * 0.95  # Leave a little headroom
            
            # Convert back to tensor and ensure correct shape
            combined_tensor = torch.from_numpy(combined).unsqueeze(0)
            
            # Return in the same format as input
            if original_format1 == 'dict':
                # Copy the original dict and update with the processed tensor
                result = audio1.copy()
                
                # Update the appropriate key
                if 'samples' in audio1:
                    result['samples'] = combined_tensor
                elif 'waveform' in audio1:
                    result['waveform'] = combined_tensor
                elif 'audio' in audio1:
                    result['audio'] = combined_tensor
                elif 'tensor' in audio1:
                    result['tensor'] = combined_tensor
                else:
                    # If we got here, we used some other key earlier
                    for key in audio1.keys():
                        if isinstance(audio1[key], torch.Tensor):
                            result[key] = combined_tensor
                            break
                
                return (result,)
            else:
                # Return as a standard dict with 'samples' key
                return ({"samples": combined_tensor},)
        except Exception as e:
            import traceback
            print(f"MasteringCombinerNode error: {str(e)}")
            print(f"Audio1 type: {type(audio1)}, Audio2 type: {type(audio2)}")
            if isinstance(audio1, dict):
                print(f"Audio1 keys: {list(audio1.keys())}")
            if isinstance(audio2, dict):
                print(f"Audio2 keys: {list(audio2.keys())}")
            traceback.print_exc()
            raise ValueError(f"MasteringCombinerNode error: {str(e)}")
            return None