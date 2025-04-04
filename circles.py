import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk, ellipse
import os

# Parameters
IMAGE_SIZE = (2400, 2400)  # Size of the generated images
NUM_IMAGES = 100        # Number of images to generate
NUM_DOTS_RANGE = (5, 20) # Range for the number of dots (cross-sections) per image
OUTPUT_DIR = "./datasets"  # Base directory to save images

# Create output directories if they don't exist
folders = ["circles_only", "circles_and_ellipses", "uniform_radius_circles", "variable_radius_per_image"]
for folder in folders:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

def generate_circle(shape, radius, location):
    """Generates a circle on a blank image."""
    img = np.zeros(shape, dtype=np.float32)
    rr, cc = disk(location, radius, shape=img.shape)
    img[rr, cc] = 1
    return img

def is_overlap(img, new_shape):
    """Check if the new shape overlaps with the existing image."""
    return np.any((img + new_shape) > 1)

def generate_random_circle(shape, existing_img, radius=None):
    """Generates a random circle with an optional fixed radius, avoiding overlap."""
    for _ in range(100):
        if radius is None:
            radius = np.random.randint(5, min(shape) // 4)
        location = (np.random.randint(radius, shape[0] - radius), np.random.randint(radius, shape[1] - radius))
        new_shape = generate_circle(shape, radius, location)
        if not is_overlap(existing_img, new_shape):
            return new_shape
    return np.zeros(shape)

def generate_dataset(folder, image_size, num_dots_range, circle_only=False, uniform_radius=None, variable_radius_per_image=False):
    """Generates a dataset of images."""
    for i in range(NUM_IMAGES):
        img = np.zeros(image_size, dtype=np.float32)
        num_dots = np.random.randint(num_dots_range[0], num_dots_range[1])
        fixed_radius = np.random.randint(5, min(image_size) // 4) if variable_radius_per_image else None

        for _ in range(num_dots):
            if circle_only:
                new_shape = generate_random_circle(image_size, img)
            elif uniform_radius is not None:
                new_shape = generate_random_circle(image_size, img, radius=uniform_radius)
            elif variable_radius_per_image:
                new_shape = generate_random_circle(image_size, img, radius=fixed_radius)
            else:
                new_shape = generate_random_circle(image_size, img)  # Default random circle for `circles_and_ellipses`
            img += new_shape

        img = np.clip(img, 0, 1)
        plt.imsave(os.path.join(OUTPUT_DIR, folder, f"cross_section_{i:04d}.png"), img, cmap='gray')

if __name__ == "__main__":
    print("Generating dataset for 'circles_only'...")
    generate_dataset("circles_only", IMAGE_SIZE, NUM_DOTS_RANGE, circle_only=True)
    
    print("Generating dataset for 'circles_and_ellipses'...")
    generate_dataset("circles_and_ellipses", IMAGE_SIZE, NUM_DOTS_RANGE)
    
    print("Generating dataset for 'uniform_radius_circles' with radius=10...")
    generate_dataset("uniform_radius_circles", IMAGE_SIZE, NUM_DOTS_RANGE, uniform_radius=10)
    
    print("Generating dataset for 'variable_radius_per_image'...")
    generate_dataset("variable_radius_per_image", IMAGE_SIZE, NUM_DOTS_RANGE, variable_radius_per_image=True)
    
    print(f"Datasets generated and saved to {OUTPUT_DIR}.")
