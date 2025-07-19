# __init__.py
from .vocal_compressor import VocalCompressorNode
from .vocal_deesser import VocalDeesserNode
from .vocal_eq import VocalEQNode
from .vocal_doubler import VocalDoublerNode
from .vocal_reverb import VocalReverbNode
from .vocal_limiter import VocalLimiterNode
from .vocal_processor import VocalProcessorNode
from .mastering_combiner import MasteringCombinerNode

# Try to import optional modules
try:
    from .mzikart_player import MzikartPlayerNode, setup_audio_routes
    HAS_PLAYER = True
except ImportError:
    print("Warning: MzikartPlayerNode could not be imported")
    HAS_PLAYER = False

try:
    from .noise_cutter import MzikartNoiseCutter
    HAS_NOISE_CUTTER = True
except ImportError as e:
    print(f"Warning: MzikartNoiseCutter could not be imported: {e}")
    print("To use noise reduction, install the required package: pip install noisereduce")
    HAS_NOISE_CUTTER = False

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VocalCompressorNode": VocalCompressorNode,
    "VocalDeesserNode": VocalDeesserNode,
    "VocalEQNode": VocalEQNode,
    "VocalDoublerNode": VocalDoublerNode,
    "VocalReverbNode": VocalReverbNode,
    "VocalLimiterNode": VocalLimiterNode,
    "VocalProcessorNode": VocalProcessorNode,
    "MasteringCombinerNode": MasteringCombinerNode
}

# Add optional nodes if available
if HAS_PLAYER:
    NODE_CLASS_MAPPINGS["MzikartPlayerNode"] = MzikartPlayerNode

if HAS_NOISE_CUTTER:
    NODE_CLASS_MAPPINGS["MzikartNoiseCutter"] = MzikartNoiseCutter

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VocalCompressorNode": "🎤 Vocal Compressor",
    "VocalDeesserNode": "🎤 Vocal De-Esser",
    "VocalEQNode": "🎤 Vocal EQ",
    "VocalDoublerNode": "🎤 Vocal Doubler",
    "VocalReverbNode": "🎤 Vocal Reverb",
    "VocalLimiterNode": "🎤 Vocal Limiter",
    "VocalProcessorNode": "🎤 Vocal Processor",
    "MasteringCombinerNode": "🔊 Mastering Combiner"
}

# Add optional display names if available
if HAS_PLAYER:
    NODE_DISPLAY_NAME_MAPPINGS["MzikartPlayerNode"] = "🎵 Mzikart Player"

if HAS_NOISE_CUTTER:
    NODE_DISPLAY_NAME_MAPPINGS["MzikartNoiseCutter"] = "🔇 Noise Cutter"

# Web directory for any JS/CSS files
WEB_DIRECTORY = "./js"

# Export only what's available
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
if HAS_PLAYER:
    __all__.append('setup_audio_routes')