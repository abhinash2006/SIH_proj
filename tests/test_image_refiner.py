import numpy as np
from src.image_refiner import ImageRefiner

def test_image_refiner_preserves_shape():
    refiner = ImageRefiner({"image_refinement": {"enabled": True}})
    dummy = np.random.randint(20, 220, (518, 518, 3), dtype=np.uint8)
    refined = refiner.refine_image(dummy)
    
    assert refined.shape == dummy.shape
    assert refined.dtype == np.uint8
    # Should not produce NaNs or infs
    assert np.all(np.isfinite(refined))

def test_image_refiner_contrast_enhancement():
    refiner = ImageRefiner({"image_refinement": {"enabled": True, "exposure_normalization": True}})
    # Low contrast image
    low_contrast = np.full((100, 100, 3), 100, dtype=np.uint8)
    low_contrast[20:40, 20:40] = 110
    
    refined = refiner.refine_image(low_contrast)
    assert refined.shape == low_contrast.shape
    # Std dev of refined should be higher or equal to low contrast original
    assert np.std(refined) >= np.std(low_contrast)
