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
    CATEGORY = "Mzikart/Mastering III"

    def limit(self, audio, threshold, release):
        try:
            # Handle the case where audio is a dictionary directly
            if isinstance(audio, dict):
                if 'samples' in audio:
                    audio_tensor = audio['samples']
                elif 'waveform' in audio:
                    audio_tensor = audio['waveform']
                else:
                    raise ValueError(
                        f"Audio dict must contain 'samples' or 'waveform' key")
            else:
                audio_tensor = audio

            # Convert to numpy for processing
            if isinstance(audio_tensor, torch.Tensor):
                if audio_tensor.dim() == 3:
                    audio_tensor = audio_tensor.squeeze(0)
                audio_np = audio_tensor.numpy()
            else:
                raise TypeError(
                    f"Expected tensor but got {type(audio_tensor)}")

            sample_rate = 48000

            # Process each channel
            processed = np.zeros_like(audio_np)
            for c in range(audio_np.shape[0]):
                processed[c] = self.process_channel(
                    audio_np[c], sample_rate,
                    threshold, release
                )

            # Convert back to tensor and ensure correct shape
            processed_tensor = torch.from_numpy(processed).unsqueeze(0)

            # Return in the same format as input
            if isinstance(audio, dict):
                # Copy the original dict and update samples
                result = audio.copy()
                if 'samples' in audio:
                    result['samples'] = processed_tensor
                elif 'waveform' in audio:
                    result['waveform'] = processed_tensor
                return (result,)
            else:
                return ({"samples": processed_tensor},)
        except Exception as e:
            raise ValueError(f"VocalLimiterNode error: {str(e)}")
            return None

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
