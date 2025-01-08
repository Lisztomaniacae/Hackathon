from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
from outputParser import denormalize_predictions
from outputParser import overlay_images
import torch.nn as nn
import torch.nn.functional as F

resize_constant = 128


class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()

        # Downsampling Path with Strided Convolutions
        self.enc1 = self._conv_block(1, resize_constant // 16, stride=2)
        self.enc2 = self._conv_block(resize_constant // 16, resize_constant // 8, stride=2)
        self.enc3 = self._conv_block(resize_constant // 8, resize_constant // 4, stride=2)

        # Bottleneck with SE Block
        self.bottleneck = self._conv_block(resize_constant // 4, resize_constant)
        self.se_block = self._squeeze_excitation_block(resize_constant)

        # Upsampling Path with Bilinear Interpolation and Attention
        self.dec3 = self._conv_block(resize_constant, resize_constant // 4)
        self.dec2 = self._conv_block(resize_constant // 4, resize_constant // 8)
        self.dec1 = self._conv_block(resize_constant // 8, resize_constant // 16)
        self.spatial_attention = self._spatial_attention_block()

        # Final Convolution
        self.final_conv = nn.Conv2d(resize_constant // 16, 1, kernel_size=1)

        # Residual Block for Machined Part and Gripper
        self.resnet_block = self._residual_block(1)

        # Adaptive Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((resize_constant // 2, 1))

        # Fully Connected Layers
        self.shared_fc1 = nn.Linear(resize_constant * 4, resize_constant // 2)
        self.shared_fc2 = nn.Linear(resize_constant // 2, resize_constant // 4)
        self.shared_fc_fin = nn.Linear(resize_constant // 4, resize_constant // 8)
        self.fc3a = nn.Linear(resize_constant // 8, 1)
        self.fc3b = nn.Linear(resize_constant // 8, 1)
        self.fc3c = nn.Linear(resize_constant * 4, resize_constant)
        self.fc4c = nn.Linear(resize_constant, resize_constant // 2)
        self.fc5c = nn.Linear(resize_constant // 2, resize_constant // 4)
        self.fc6c = nn.Linear(resize_constant // 4, resize_constant // 8)
        self.fc7c = nn.Linear(resize_constant // 8, 1)

        self._initialize_weights()

    def _conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _squeeze_excitation_block(self, channels, reduction=16):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def _spatial_attention_block(self):
        return nn.Sequential(
            nn.Conv2d(resize_constant // 2, 1, kernel_size=7, padding=3),  # Change input channels to 32
            nn.Sigmoid()
        )

    def _residual_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.5)

    def shift_gripper(self, gripper, a, b):
        """
        Shifts the pixels of the `gripper` tensor based on parameters `a` and `b`.

        Args:
            gripper: Tensor of shape (B, 1, H, W), where B is the batch size.
            a: Tensor of shape (B,) with values in [0, 1], controlling horizontal shift.
            b: Tensor of shape (B,) with values in [0, 1], controlling vertical shift.

        Returns:
            Shifted gripper tensor of the same shape as the input.
        """
        B, _, H, W = gripper.shape

        # Calculate shifts in pixels
        dx = ((a - 0.5) * W).long()  # Horizontal shift
        dy = ((b - 0.5) * H).long()  # Vertical shift

        shifted_gripper = torch.zeros_like(gripper)

        for i in range(B):
            x_shift = dx[i].item()
            y_shift = dy[i].item()

            # Compute source slicing bounds
            src_x_start = max(0, -x_shift)
            src_x_end = min(W, W - x_shift)
            src_y_start = max(0, -y_shift)
            src_y_end = min(H, H - y_shift)

            # Compute destination slicing bounds
            dst_x_start = max(0, x_shift)
            dst_x_end = min(W, W + x_shift)
            dst_y_start = max(0, y_shift)
            dst_y_end = min(H, H + y_shift)

            # Ensure that source and destination slice sizes match
            if (src_x_end > src_x_start) and (src_y_end > src_y_start):
                shifted_gripper[i, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = gripper[
                                                                                      i, :, src_y_start:src_y_end,
                                                                                      src_x_start:src_x_end
                                                                                      ]

        return shifted_gripper

    def unet(self, x):
        # Downsampling
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)
        bottleneck = bottleneck * self.se_block(bottleneck)

        # Upsampling with Attention
        dec3 = self.dec3(bottleneck)
        dec3 = dec3 * self.spatial_attention(torch.cat([enc3, dec3], dim=1))

        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)

        # Final Convolution
        output = self.final_conv(dec1)
        return output

    def forward(self, x):
        machined_part = x[:, 0, :, :].unsqueeze(1)
        gripper = x[:, 1, :, :].unsqueeze(1)

        # Process machined part and unmodified gripper through UNet
        unet_output_machined = self.unet(machined_part)
        unet_output_gripper = self.unet(gripper)

        # Residual Connection
        residual_machined = self.resnet_block(unet_output_machined)
        residual_gripper = self.resnet_block(unet_output_gripper)

        residual_output = torch.cat((residual_machined, residual_gripper), dim=1)

        # Adaptive Average Pooling and Flattening
        pooled_output = residual_output.flatten(1)

        # Shared Fully Connected Layers for a and b
        shared = F.relu(self.shared_fc1(pooled_output))
        shared = F.relu(self.shared_fc2(shared))
        shared = F.relu(self.shared_fc_fin(shared))

        a = self.fc3a(shared)
        b = self.fc3b(shared)

        # Shift Gripper Based on a and b
        shifted_gripper = self.shift_gripper(gripper, a, b)

        # Process shifted gripper through UNet
        unet_output_shifted_gripper = self.unet(shifted_gripper)

        # Combine UNet Outputs and Pass to c
        combined_unet_output = torch.cat([unet_output_machined, unet_output_shifted_gripper], dim=1).flatten(1)
        c = F.relu(self.fc3c(combined_unet_output))
        c = F.relu(self.fc4c(c))
        c = F.relu(self.fc5c(c))
        c = F.relu(self.fc6c(c))
        c = self.fc7c(c)

        # Concatenate a, b, c
        output = torch.cat([a, b, c], dim=1)
        return output


def init():
    # Initialize the model architecture
    global model
    global device
    directory = Path(__file__).parent.parent
    model = ConvolutionalNeuralNet()
    # Load the saved state dict into the model
    model.load_state_dict(torch.load(directory / 'doc/cnn.pth', map_location="cpu"))

    # Move the model to the appropriate device
    device = torch.device("cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


def forward(input_tensor, bias_x, bias_y, norm_x, norm_y, model):
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model.forward(input_tensor).squeeze(0)
    x, y, alpha = denormalize_predictions(prediction, bias_x, bias_y, norm_x, norm_y)
    return x, y, alpha


def visualize(machined_part_path, gripper_path, x, y, alpha):
    # Open the images using PIL
    machined_part = Image.open(machined_part_path)
    gripper = Image.open(gripper_path)

    # Overlay the gripper on the machined part
    overlaid_image = overlay_images(machined_part, gripper, x, y, alpha)

    # Display the result using matplotlib
    plt.imshow(overlaid_image)
    plt.axis('off')  # Hide the axes
    plt.show()
