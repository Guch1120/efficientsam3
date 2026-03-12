"""EfficientSAM compatibility package.

Use this package name to avoid conflicts with upstream `sam3` modules in ROS workspaces.
"""

from importlib import import_module

# Re-export main builders from the nested implementation package.
build_sam3_image_model = import_module("sam3.sam3.model_builder").build_sam3_image_model
build_efficientsam3_image_model = import_module("sam3.sam3.model_builder").build_efficientsam3_image_model

__all__ = [
    "build_sam3_image_model",
    "build_efficientsam3_image_model",
]
