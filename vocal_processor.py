import torch
import numpy as np
from .vocal_compressor import VocalCompressorNode
from .vocal_deesser import VocalDeesserNode
from .vocal_eq import VocalEQNode
from .vocal_doubler import VocalDoublerNode
from .vocal_reverb import VocalReverbNode
from .vocal_limiter import VocalLimiterNode


class VocalProcessorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "genre": (["general", "rap", "rnb", "gospel", "pop"], {"default": "general"}),
                "aggressiveness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("processed_vocals",)
    FUNCTION = "process_vocals"
    CATEGORY = "Mzikart/Mastering"

    def process_vocals(self, audio, genre, aggressiveness):
        # Ensure audio is properly formatted
        if not isinstance(audio, dict) and not isinstance(audio, torch.Tensor):
            raise TypeError(
                f"Expected audio to be dict or torch.Tensor but got {type(audio)}")

        # Apply genre-specific presets
        if genre == "rap":
            comp_threshold = -20.0 + (aggressiveness * -5.0)
            comp_ratio = 4.0 + (aggressiveness * 2.0)
            eq_air = 2.0 + (aggressiveness * 2.0)
        elif genre == "rnb":
            comp_threshold = -25.0
            comp_ratio = 3.0
            eq_air = 4.0
        elif genre == "gospel":
            comp_threshold = -22.0
            comp_ratio = 2.5
            eq_air = 3.0
        else:  # pop/general
            comp_threshold = -23.0
            comp_ratio = 3.5
            eq_air = 5.0

        # Create processing chain
        comp = VocalCompressorNode()
        deess = VocalDeesserNode()
        eq = VocalEQNode()
        doubler = VocalDoublerNode()
        reverb = VocalReverbNode()
        limiter = VocalLimiterNode()

        # Process through chain
        compressed = comp.compress(
            audio, comp_threshold, comp_ratio, 10.0, 150.0, 4.0, 6.0, genre)[0]
        deessed = deess.deess(compressed, 0.7, 6000.0)[0]
        eqed = eq.apply_eq(deessed, 80.0, 3.0, 3000.0, eq_air, 12000.0)[0]
        doubled = doubler.double(eqed, 0.6, 0.02, 0.03)[0]
        reverbed = reverb.apply_reverb(doubled, "hall", 1.5, 0.15)[0]
        limited = limiter.limit(reverbed, -0.3, 30.0)[0]

        # Return in the same format as input
        if isinstance(audio, dict) and 'samples' in audio:
            # Copy the original dict and update samples
            if isinstance(limited, dict) and 'samples' in limited:
                result = audio.copy()
                result['samples'] = limited['samples']
                return (result,)
            else:
                result = audio.copy()
                result['samples'] = limited
                return (result,)
        else:
            if isinstance(limited, dict) and 'samples' in limited:
                return (limited,)
            else:
                return ({"samples": limited},)
