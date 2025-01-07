from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from rich.progress import track
import pandas as pd
import CNN
from inputParser import parse_images_to_input_tensor


def compute_solution(
        part_image_path: Path, gripper_image_path: Path, model
) -> tuple[float, float, float]:
    """Compute the solution for the given part and gripper images.

    :param part_image_path: Path to the part image
    :param gripper_image_path: Path to the gripper image
    :return: The x, y and angle of the gripper
    """
    input_tensor, bias_x, bias_y, norm_x, norm_y = parse_images_to_input_tensor(part_image_path, gripper_image_path)
    x, y, alpha = CNN.forward(input_tensor, bias_x, bias_y, norm_x, norm_y, model)
    CNN.visualize(part_image_path, gripper_image_path, x, y, alpha)
    angle = np.degrees((alpha + 2 * np.pi) % (2 * np.pi))
    return x, y, angle


def main():
    """The main function of your solution.

    Feel free to change it, as long as it maintains the same interface.
    """
    model = CNN.init()
    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    # read the input csv file
    input_df = pd.read_csv(args.input)

    # compute the solution for each row
    results = []
    for _, row in track(
            input_df.iterrows(),
            description="Computing the solutions for each row",
            total=len(input_df),
    ):
        try:
            part_image_path = Path(str(row["part"]))
            gripper_image_path = Path(str(row["gripper"]))
            assert part_image_path.exists(), f"{part_image_path} does not exist"
            assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
            x, y, angle = compute_solution(part_image_path, gripper_image_path, model)
            results.append([str(part_image_path), str(gripper_image_path), x, y, angle])
        except:
            continue

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
