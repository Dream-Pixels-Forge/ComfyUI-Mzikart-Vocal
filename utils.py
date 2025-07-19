import torch
import numpy as np

def extract_audio_tensor(audio):
    """Extract audio tensor from various input formats.
    
    Args:
        audio: Can be a torch.Tensor or a dict with 'samples' or other keys
        
    Returns:
        tuple: (tensor, sample_rate, original_format)
    """
    sample_rate = 48000  # Default sample rate
    original_format = None
    
    try:
        if isinstance(audio, dict):
            original_format = 'dict'
            # Try common keys for audio data
            if 'samples' in audio:
                tensor = audio['samples']
            elif 'waveform' in audio:
                tensor = audio['waveform']
            elif 'audio' in audio:
                tensor = audio['audio']
            elif 'tensor' in audio:
                tensor = audio['tensor']
            else:
                # Print available keys for debugging
                keys = list(audio.keys())
                print(f"Available keys in audio dict: {keys}")
                # Try the first key that might contain a tensor
                for key in audio.keys():
                    if isinstance(audio[key], torch.Tensor):
                        print(f"Using key '{key}' which contains a tensor")
                        tensor = audio[key]
                        break
                else:
                    raise ValueError(f"Could not find tensor in audio dict with keys: {keys}")
            
            # Check if sample_rate is provided
            if 'sample_rate' in audio:
                sample_rate = audio['sample_rate']
        elif isinstance(audio, torch.Tensor):
            original_format = 'tensor'
            tensor = audio
        else:
            raise TypeError(f"Expected audio to be dict or torch.Tensor but got {type(audio)}")
        
        # Ensure tensor is properly formatted
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor to be torch.Tensor but got {type(tensor)}")
            
        return tensor, sample_rate, original_format
    
    except Exception as e:
        import traceback
        print(f"Error extracting audio tensor: {str(e)}")
        print(f"Audio type: {type(audio)}")
        if isinstance(audio, dict):
            print(f"Audio keys: {list(audio.keys())}")
        traceback.print_exc()
        raise