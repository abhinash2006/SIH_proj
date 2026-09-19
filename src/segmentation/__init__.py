"""Segmentation module for pixel-accurate dynamic object masking."""

from .sam2_segmenter import SAM2Segmenter, MaskResult

__all__ = ["SAM2Segmenter", "MaskResult"]
