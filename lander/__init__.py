"""Vision-based autonomous landing on a moving platform with PPO."""
import os

# Must be set before anything imports google.protobuf (see env.sh for why).
# setdefault: respect an explicit choice made by the caller's environment.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

__version__ = "0.0.1"
