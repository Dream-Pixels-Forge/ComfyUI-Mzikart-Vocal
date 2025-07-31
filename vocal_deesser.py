import torch
import numpy as np
from scipy.signal import butter, sosfilt
from .utils import extract_audio_tensor


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
    CATEGORY = "Mzikart/Mastering III"

    def deess(self, audio, sensitivity, frequency):
        try:
            # Extract audio tensor using the helper function
            audio_tensor, sample_rate, original_format = extract_audio_tensor(
                audio)

            # Convert to numpy for processing
            if audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)
            audio_np = audio_tensor.numpy()

            # Create band-pass filter for sibilance range
            sos = butter(4, [frequency-1000, frequency+1000],
                         'bandpass', fs=sample_rate, output='sos')

            # Process each channel
            processed = np.zeros_like(audio_np)
            for c in range(audio_np.shape[0]):
                processed[c] = self.process_channel(
                    audio_np[c], sample_rate, sos, sensitivity
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
            print(f"VocalDeesserNode error: {str(e)}")
            print(f"Audio type: {type(audio)}")
            if isinstance(audio, dict):
                print(f"Audio keys: {list(audio.keys())}")
            traceback.print_exc()
            raise ValueError(f"VocalDeesserNode error: {str(e)}")
            return None

    def process_channel(self, audio, sample_rate, sos, sensitivity):
        # Extract sibilance frequencies
        sibilance = sosfilt(sos, audio)

        # Detect sibilance peaks
        rms = np.sqrt(np.convolve(sibilance**2, np.ones(50)/50, mode='same'))
        threshold = np.percentile(rms, 95) * sensitivity
        reduction = np.where(rms > threshold, threshold / (rms + 1e-7), 1.0)

        # Apply reduction only to sibilance range
        return audio * reduction
