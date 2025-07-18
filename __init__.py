# __init__.py
from .vocal_compressor import VocalCompressorNode
from .vocal_deesser import VocalDeesserNode
from .vocal_eq import VocalEQNode
from .vocal_doubler import VocalDoublerNode
from .vocal_reverb import VocalReverbNode
from .vocal_limiter import VocalLimiterNode
from .vocal_processor import VocalProcessorNode

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VocalCompressorNode": VocalCompressorNode,
    "VocalDeesserNode": VocalDeesserNode,
    "VocalEQNode": VocalEQNode,
    "VocalDoublerNode": VocalDoublerNode,
    "VocalReverbNode": VocalReverbNode,
    "VocalLimiterNode": VocalLimiterNode,
    "VocalProcessorNode": VocalProcessorNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VocalCompressorNode": "🎤 Vocal Compressor",
    "VocalDeesserNode": "🎤 Vocal De-Esser",
    "VocalEQNode": "🎤 Vocal EQ",
    "VocalDoublerNode": "🎤 Vocal Doubler",
    "VocalReverbNode": "🎤 Vocal Reverb",
    "VocalLimiterNode": "🎤 Vocal Limiter",
    "VocalProcessorNode": "🎤 Vocal Processor"
}

# Web directory for any JS/CSS files
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']