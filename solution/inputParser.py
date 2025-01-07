# Importing dependencies
import torch

import cv2
import numpy as np
from PIL import Image

resize_constant = 128


def remove_black_borders_and_track_bias(image):
    """
    Removes black rows and columns from a grayscale image, tracks x and y biases,
    and adds random padding around the image.

    Args:
        image (numpy.ndarray): Grayscale image to process.

    Returns:
        numpy.ndarray: Padded image without black borders.
        float: x_bias
        float: y_bias
    """
    x_bias, y_bias = 0, 0
    rows, cols = image.shape

    # Remove rows from the top
    while rows > 0 and np.all(image[0, :] == 0):  # Assuming black is 0
        image = image[1:, :]  # Remove the top row
        y_bias += 0.5
        rows -= 1

    # Remove rows from the bottom
    while rows > 0 and np.all(image[-1, :] == 0):
        image = image[:-1, :]  # Remove the bottom row
        y_bias -= 0.5
        rows -= 1

    # Remove columns from the left
    while cols > 0 and np.all(image[:, 0] == 0):
        image = image[:, 1:]  # Remove the left column
        x_bias += 0.5
        cols -= 1

    # Remove columns from the right
    while cols > 0 and np.all(image[:, -1] == 0):
        image = image[:, :-1]  # Remove the right column
        x_bias -= 0.5
        cols -= 1

    return image, x_bias, y_bias


def pad_and_resize(img, target_pad_size, target_size=(resize_constant, resize_constant)):
    """
    Pads the image to the target size times 8 with black pixels and resizes it to target size.
    Args:
        img (numpy.ndarray): Input grayscale image.
        target_size (tuple): Desired size (width, height).
    Returns:
        numpy.ndarray: Resized image.
    """

    # First pad the image to maintain aspect ratio
    target_pad_width, target_pad_height = target_pad_size
    current_height, current_width = img.shape

    # Calculate padding
    pad_width = max(target_pad_width - current_width, 0)
    pad_height = max(target_pad_height - current_height, 0)

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Resize the padded image to target size
    resized_img = cv2.resize(padded_img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img


def parse_images_to_input_tensor(machined_part_path, gripper_path):
    # Load images using PIL
    machined_part = Image.open(machined_part_path).convert("RGBA")

    # Convert to numpy array (RGBA)
    machined_part = np.array(machined_part)

    # Subtract 2, Add 1 to all RGB channels, ensure values stay within 0-255
    machined_part[:, :, :3] = np.clip(machined_part[:, :, :3] - 2, 0, 255)
    machined_part[:, :, :3] = np.clip(machined_part[:, :, :3] + 1, 0, 255)

    # Replace transparent pixels with black
    # Pixels where alpha channel == 0
    machined_part[machined_part[:, :, 3] == 0] = [0, 0, 0, 255]

    # Convert RGBA to RGB (drop alpha channel)
    machined_part = cv2.cvtColor(machined_part, cv2.COLOR_RGBA2RGB)

    # Convert to grayscale
    machined_part = cv2.cvtColor(machined_part, cv2.COLOR_RGB2GRAY)
    gripper = cv2.imread(gripper_path, cv2.IMREAD_GRAYSCALE)
    if gripper is None:
        print(gripper_path)
        raise ValueError("No Gripper bruh")

    # Crop blackspace of machined part
    machined_part, bias_x, bias_y = remove_black_borders_and_track_bias(machined_part)

    bias_x, bias_y = 0, 0

    width_part, height_part = machined_part.shape[1], machined_part.shape[0]
    width_gripper, height_gripper = gripper.shape[1], gripper.shape[0]

    norm_x = max(width_part, width_gripper)
    norm_y = max(height_part, height_gripper)

    norm_x = max(norm_x, norm_y)
    norm_y = norm_x

    # Resize images with padding
    machined_part = pad_and_resize(machined_part, (norm_x, norm_y))
    gripper = pad_and_resize(gripper, (norm_x, norm_y))

    if len(machined_part[machined_part > 0]) == 0:
        print(machined_part_path)
        raise ValueError("Image contains no non-zero pixels")

    combined_images = np.stack((machined_part, gripper), axis=0)  # Shape: (2, resize_constant, resize_constant)
    combined_images = combined_images / 255.0  # Normalize pixel values to [0, 1]

    # Combine inputs
    input_tensor = torch.tensor(combined_images, dtype=torch.float32)

    return input_tensor, bias_x, bias_y, norm_x, norm_y
