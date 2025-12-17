import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

def get_canny_image(images):
    images = np.array(images)
    canny_images = []
    if images.dtype != np.uint8:
        assert images.min() >= 0 and images.max() <= 1, "Float images should be in range [0, 1]"
        images = (images * 255).astype(np.uint8)
    for image in images:
        canny = cv2.Canny(image, 100, 200)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny_image = Image.fromarray(canny)
        canny_images.append(canny_image)
    return canny_images
    


def get_depth_estimation(images):
    """
    Args:
        images:
            - np.ndarray of shape [B, H, W, C] or [B, H, W] in [0,1] or uint8
            - OR list of PIL.Image.Image

    Returns:
        depth_pil: list of PIL.Image.Image (RGB) with per-image min-max normalization.
    """
    # 1. Normalize input to list of PIL images
    if isinstance(images, np.ndarray):
        # Expect [B, H, W] or [B, H, W, C]
        if images.dtype != np.uint8:
            assert images.min() >= 0 and images.max() <= 1, "Float images should be in range [0, 1]"
            images = (images * 255).astype(np.uint8)

        # If [B, H, W], add a channel dimension for grayscale
        if images.ndim == 3:
            images = images[..., None]

        pil_images = [Image.fromarray(img) for img in images]
    else:
        # assume already a list of PIL.Image.Image
        pil_images = images

    # 2. Run depth estimation pipeline
    checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    pipe = pipeline("depth-estimation", model=checkpoint, device="cuda")
    predictions = pipe(pil_images)
    pipe.model.to("cpu")

    # 3. Extract depth maps as PIL, then convert to normalized uint8 RGB
    depth_pils = []
    for pred in predictions:
        # Depending on the pipeline, key may be "predicted_depth" or "depth"
        depth_img = pred.get("depth", pred.get("predicted_depth"))
        if depth_img is None:
            raise KeyError("Depth prediction dict does not contain 'depth' or 'predicted_depth'.")

        # Convert to numpy
        d = np.array(depth_img).astype(np.float32)  # [H, W]

        # Min-max normalize per image to [0, 1]
        d_min = d.min()
        d_max = d.max()
        if d_max > d_min:
            d = (d - d_min) / (d_max - d_min)
        else:
            d = np.zeros_like(d)

        # To [0, 255] uint8
        d_uint8 = (d * 255.0).round().astype(np.uint8)  # [H, W]

        # Make it RGB by repeating channels: [H, W, 3]
        d_rgb = np.repeat(d_uint8[..., None], 3, axis=-1)

        depth_pils.append(Image.fromarray(d_rgb))

    return depth_pils
    