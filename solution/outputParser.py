import numpy as np
from PIL import Image


def denormalize_predictions(pred, x_bias, y_bias, norm_x, norm_y):
    x = int(np.floor((pred[0].item() - 0.5) * norm_x + x_bias))  # x in original scale
    y = int(np.floor((pred[1].item() - 0.5) * norm_y + y_bias))  # y in original scale
    alpha = (pred[2].item() - 0.5) * np.pi * 2  # alpha back to original range [-pi, pi]
    return x, y, alpha


def overlay_images(base_img, overlay_img, x, y, alpha):
    """
    Overlay the overlay_img onto base_img at the given position, with rotation (alpha).

    Arguments:
    - base_img: The base image (PIL Image) onto which the overlay will be placed.
    - overlay_img: The overlay image (PIL Image) that will be placed on the base image.
    - x, y: Coordinates for placing the overlay image, with (0, 0) representing the center.
      Positive x is to the right, negative x is to the left.
      Positive y is upwards, negative y is downwards.
    - alpha: Rotation angle in radians for the overlay image.

    Returns:
    - The base image with the overlay placed on top at the specified coordinates and rotation.
    """
    # Ensure the overlay image is in RGBA mode (with transparency)
    overlay_img = overlay_img.convert("RGBA")

    # Rotate the overlay image by the specified angle (alpha in radians)
    overlay_img = overlay_img.rotate(np.degrees(-alpha), resample=Image.Resampling.BICUBIC, expand=True)

    # Get the dimensions of the rotated overlay image
    overlay_w, overlay_h = overlay_img.size

    # Get the dimensions of the base image
    base_w, base_h = base_img.size

    # Compute the position (center the overlay on (x, y))
    top_left_x = int(base_w // 2 - overlay_w // 2 + x)
    top_left_y = int(base_h // 2 - overlay_h // 2 + y)

    # Create a transparent background (RGBA) canvas
    canvas = Image.new("RGBA", base_img.size, (0, 0, 0, 0))  # Transparent background

    # Paste the base image onto the canvas
    canvas.paste(base_img.convert("RGBA"), (0, 0))

    # Paste the overlay image onto the canvas at the calculated position
    # The mask ensures that the transparency (alpha) is respected
    canvas.paste(overlay_img, (top_left_x, top_left_y), overlay_img)

    # Return the result (optional: you can return as "RGB" to remove transparency)
    return canvas
