"""Independent SAMGA + visual-LoRA experiment implementation."""

from .model import SAMGATaskModel, MultiLayerCLIPProvider, SAMGALoss

__all__ = ["SAMGATaskModel", "MultiLayerCLIPProvider", "SAMGALoss"]
