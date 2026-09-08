import numpy as np

class DepthConfidenceFused:
    """
    Fuses VGGT multi-view geometric confidence with Depth Anything V2 monocular consistency maps:
        Final_Confidence = w_vggt * VGGT_Confidence + w_da2 * Consistency_Map
    Provides robust confidence estimates for 3D point cloud filtering.
    """

    def __init__(self, vggt_weight: float = 0.6, da2_weight: float = 0.4):
        self.vggt_weight = vggt_weight
        self.da2_weight = da2_weight

    def fuse_confidence(
        self,
        vggt_conf: np.ndarray,
        consistency_map: np.ndarray
    ) -> np.ndarray:
        """
        Fuses confidence maps into a single spatial quality map [H, W].
        """
        if vggt_conf.ndim == 3:
            vggt_conf = vggt_conf.squeeze(-1)
        if consistency_map.ndim == 3:
            consistency_map = consistency_map.squeeze(-1)

        # Normalize VGGT confidence to roughly [0, 1] range if needed
        conf_min, conf_max = vggt_conf.min(), vggt_conf.max()
        if conf_max > conf_min:
            norm_vggt_conf = (vggt_conf - conf_min) / (conf_max - conf_min)
        else:
            norm_vggt_conf = np.ones_like(vggt_conf)

        fused_conf = (self.vggt_weight * norm_vggt_conf) + (self.da2_weight * consistency_map)
        return fused_conf.astype(np.float32)
