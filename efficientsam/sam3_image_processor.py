"""Re-export image processor under `efficientsam` namespace."""

from sam3.sam3.model.sam3_image_processor import Sam3Processor

__all__ = ["Sam3Processor"]
