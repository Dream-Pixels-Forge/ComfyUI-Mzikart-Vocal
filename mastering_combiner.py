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
    CATEGORY = "Mzikart/Mastering III"

    def combine_audio(self, audio1, audio2, mix_ratio, normalize):
        try:
            # Extract audio tensors using the helper function
            audio1_tensor, sample_rate1, original_format1 = extract_audio_tensor(
                audio1)
            audio2_tensor, sample_rate2, original_format2 = extract_audio_tensor(
                audio2)

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

                # Resize arrays with proper dimensions
                audio1_resized = np.zeros((channels, length))
                audio2_resized = np.zeros((channels, length))

                # Copy data from audio1 - handle different channel counts
                for c in range(min(channels, audio1_np.shape[0])):
                    # Get the slice length for this channel
                    slice_length = min(length, audio1_np.shape[1])
                    # Copy the data
                    audio1_resized[c,
                                   :slice_length] = audio1_np[c, :slice_length]

                # If audio1 has fewer channels than the target, duplicate the last channel
                if audio1_np.shape[0] < channels:
                    for c in range(audio1_np.shape[0], channels):
                        audio1_resized[c] = audio1_resized[0]

                # Copy data from audio2 - handle different channel counts
                for c in range(min(channels, audio2_np.shape[0])):
                    # Get the slice length for this channel
                    slice_length = min(length, audio2_np.shape[1])
                    # Copy the data
                    audio2_resized[c,
                                   :slice_length] = audio2_np[c, :slice_length]

                # If audio2 has fewer channels than the target, duplicate the last channel
                if audio2_np.shape[0] < channels:
                    for c in range(audio2_np.shape[0], channels):
                        audio2_resized[c] = audio2_resized[0]

                audio1_np = audio1_resized
                audio2_np = audio2_resized

            # Mix the audio - ensure proper broadcasting
            # First make sure both arrays have the same shape
            if audio1_np.shape != audio2_np.shape:
                print(
                    f"Warning: Audio shapes still don't match after resizing: {audio1_np.shape} vs {audio2_np.shape}")
                # Use the smaller shape for both
                min_channels = min(audio1_np.shape[0], audio2_np.shape[0])
                min_length = min(audio1_np.shape[1], audio2_np.shape[1])
                audio1_np = audio1_np[:min_channels, :min_length]
                audio2_np = audio2_np[:min_channels, :min_length]

            # Now mix the audio
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

            # Print detailed shape information for debugging
            if 'audio1_np' in locals() and 'audio2_np' in locals():
                print(
                    f"Audio1 shape: {audio1_np.shape}, Audio2 shape: {audio2_np.shape}")
                print(
                    f"Audio1 channels: {audio1_np.shape[0]}, Audio2 channels: {audio2_np.shape[0]}")
                print(
                    f"Audio1 length: {audio1_np.shape[1]}, Audio2 length: {audio2_np.shape[1]}")

            # Print tensor information
            if 'audio1_tensor' in locals() and 'audio2_tensor' in locals():
                print(f"Original audio1_tensor shape: {audio1_tensor.shape}")
                print(f"Original audio2_tensor shape: {audio2_tensor.shape}")

            traceback.print_exc()
            raise ValueError(f"MasteringCombinerNode error: {str(e)}")
            return None
