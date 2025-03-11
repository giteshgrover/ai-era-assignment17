import torch
import numpy as np
import cv2
import torch.nn.functional as F

# Function to calculate focus (sharpness) using the Laplacian variance
def calculate_focus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)     # Apply Laplacian filter
    variance = laplacian.var()                       # Compute variance (sharpness)
    return variance

# Function to calculate the blur in the background
def calculate_background_blur(image, center_x, center_y, radius):
    # Create a circular mask around the center to isolate the background
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # Create circular mask
    
    # Focus measure for background (outside the circle)
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    
    # Measure the focus (sharpness) of the background (outside the mask)
    blur_measurement = calculate_focus(background)
    return blur_measurement

# Function to check if the image is centered focus and blur in the background
def check_focus_and_blur(image):
    # Find the center of the image
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Calculate focus at the center (sharpness in the center)
    center_focus = calculate_focus(image)
    
    # Calculate blur in the background (sharpness outside the center)
    radius = min(width, height) // 4  # Set a reasonable radius to cover the center area
    background_blur = calculate_background_blur(image, center_x, center_y, radius)
    
    # Threshold to determine if it's in focus (can adjust as per requirements)
    threshold_focus = 100  # Example threshold value for center focus sharpness
    threshold_blur = 50    # Example threshold for background blur sharpness
    
    # Check if the image has a focused center and blurred background
    if center_focus > threshold_focus and background_blur < threshold_blur:
        return 0  # Good image (no loss)
    else:
        return 1  # Bad image (loss)

# Function to calculate the loss using PyTorch tensor image
def calculate_focus_blur_loss(image):
    # Convert tensor (C, H, W) to numpy (H, W, C) for OpenCV processing
    # image = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    image = image.clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")

    # Get focus and blur check result
    loss_value = check_focus_and_blur(image)

    # Convert loss_value into a PyTorch tensor
    loss = torch.tensor(loss_value, dtype=torch.float32)

    return loss