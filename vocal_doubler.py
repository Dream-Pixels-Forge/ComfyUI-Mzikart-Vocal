import torch
import numpy as np
import random
from . import extract_audio_tensor


class VocalDoublerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "intensity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "delay": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005}),
                "pitch_shift": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.1, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("doubled_audio",)
    FUNCTION = "double"
    CATEGORY = "Mzikart/Mastering"

    def double(self, audio, intensity, delay, pitch_shift):
        try:
            # Extract audio tensor using the helper function
            audio_tensor, sample_rate, original_format = extract_audio_tensor(
                audio)

            # Convert to numpy for processing
            if audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)
            audio_np = audio_tensor.numpy()

            # Process each channel
            processed = np.zeros_like(audio_np)
            for c in range(audio_np.shape[0]):
                processed[c] = self.process_channel(
                    audio_np[c], sample_rate,
                    intensity, delay, pitch_shift
                )

            # Convert back to tensor and ensure correct shape
            processed_tensor = torch.from_numpy(processed).unsqueeze(0)

            # Return in the same format as input
            if original_format == 'dict':
                # Copy the original dict and update with the processed tensor
                result = audio.copy()

                # Update the appropriate key
                if 'samples' in audio:
                    result['samples'] = processed_tensor
                elif 'waveform' in audio:
                    result['waveform'] = processed_tensor
                elif 'audio' in audio:
                    result['audio'] = processed_tensor
                elif 'tensor' in audio:
                    result['tensor'] = processed_tensor
                else:
                    # If we got here, we used some other key earlier
                    for key in audio.keys():
                        if isinstance(audio[key], torch.Tensor):
                            result[key] = processed_tensor
                            break

                return (result,)
            else:
                # Return as a standard dict with 'samples' key
                return ({"samples": processed_tensor},)
        except Exception as e:
            import traceback
            print(f"VocalDoublerNode error: {str(e)}")
            print(f"Audio type: {type(audio)}")
            if isinstance(audio, dict):
                print(f"Audio keys: {list(audio.keys())}")
            traceback.print_exc()
            raise ValueError(f"VocalDoublerNode error: {str(e)}")
            return None

    def process_channel(self, audio, sample_rate, intensity, delay, pitch_shift):
        # Create doubled version
        doubled = np.zeros_like(audio)

        # Apply delay
        delay_samples = int(delay * sample_rate)
        if delay_samples > 0:
            doubled[delay_samples:] = audio[:-delay_samples]

        # Apply pitch shift using granular synthesis
        grain_size = int(0.03 * sample_rate)  # 30ms grains
        for i in range(0, len(audio) - grain_size, grain_size//2):
            grain = audio[i:i+grain_size]

            # Apply slight pitch variation
            pitch_variation = pitch_shift * random.uniform(-1, 1)
            if pitch_variation > 0:
                # Speed up (higher pitch)
                new_length = int(grain_size / (1 + pitch_variation))
                grain = np.interp(
                    np.linspace(0, grain_size-1, new_length),
                    np.arange(grain_size),
                    grain
                )
            else:
                # Slow down (lower pitch)
                new_length = int(grain_size / (1 - pitch_variation))
                grain = np.interp(
                    np.linspace(0, grain_size-1, new_length),
                    np.arange(grain_size),
                    grain
                )

            # Add to doubled track with windowing
            start = min(i, len(doubled) - new_length)
            end = start + new_length
            window = np.hanning(new_length)
            doubled[start:end] += grain * window * intensity

        # Mix with original
        return 0.7 * audio + 0.3 * doubled
