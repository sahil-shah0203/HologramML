# -------------------------------
# PART A: IMPORTS & ITERATIVE CODE
# -------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -- TensorFlow (for iterative optimization)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# -- PyTorch (for model training)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

def compute_phase_mask_iterative(input_img_path: str,
                                 size_holo=1600,
                                 size_pad=300,
                                 num_layers_z=4,
                                 learning_rate=0.003,
                                 train_epochs=10):
    """
    This function performs iterative optimization to generate a phase mask.
    It:
      1) Loads an ideal intensity target image from input_img_path.
      2) Resizes it to (size_padded, size_padded) and replicates it across num_layers_z slices.
      3) Uses an Angular Spectrum Method (ASM) to propagate a phase mask.
      4) Optimizes the phase mask (initialized randomly) so that the propagated field matches the intensity target.
      5) Returns the optimized phase mask (cropped to the hologram region and normalized to [0,1]).
    
    The returned phase mask is a 2D array.
    """
    print(f"[Iterative] Processing image: {input_img_path}")
    # Setup parameters
    nm = 1e-9; um = 1e-6; mm = 1e-3
    lamb = 405 * nm            # wavelength (m)
    pixel_size = 1.5 * um
    size_padded = size_holo + 2 * size_pad

    # Load and resize target image (ideal intensity) and normalize to [0,1]
    img_pil = Image.open(input_img_path).convert('L')
    img_pil = img_pil.resize((size_padded, size_padded), Image.BICUBIC)
    img_np = np.array(img_pil, dtype=np.float32) / 255.0

    # Replicate the image over num_layers_z slices; we use one slice (e.g., the first) for loss.
    holo_target = np.stack([img_np] * num_layers_z, axis=-1)  # shape: (H, W, num_layers_z)

    # --------- ASM Preparation ---------
    dfx = 1 / (size_padded * pixel_size)
    dfy = 1 / (size_padded * pixel_size)
    k_x = dfx * (np.linspace(-size_padded/2, size_padded/2 - 1, size_padded))
    k_y = dfy * (np.linspace(-size_padded/2, size_padded/2 - 1, size_padded))
    fx, fy = np.meshgrid(k_x, -k_y)
    p = (fx**2 + fy**2) * (lamb**2)
    p = p.astype(np.float32)

    def asm_gpu(ob, z, lam, mx, my, pixel_size):
        # Precompute p as a TensorFlow constant
        p_tf = tf.constant(p)
        sp = tf.sqrt(1 - p_tf)
        sp = tf.cast(sp, tf.complex64)
        factor = 2 * np.pi * 1j * z / lam
        # q is the propagation kernel (transfer function)
        q = tf.signal.fftshift(tf.exp(factor * sp))
        return tf.ifft2d(tf.fft2d(ob) * q)

    # --------- Build TensorFlow Graph ---------
    # x1: z-plane position (a single value wrapped as a 1-element vector)
    x1 = tf.placeholder(tf.complex64, [1])
    # x2: target intensity image for one slice, size: (size_padded, size_padded)
    x2 = tf.placeholder(tf.float32, [size_padded, size_padded])
    # Random initialization of phase mask (2D, shape: [size_padded, size_padded])
    phase_init = tf.random_normal([size_padded, size_padded])
    phase_mask = tf.Variable(phase_init, dtype=tf.float32)
    # Enforce phase values into [0, 2π) using modulo arithmetic:
    phase_mask = 2 * np.pi * (tf.mod(phase_mask, 1))
    # The phase mask is now a 2D array that will be optimized.
    # Convert phase mask into a complex field (magnitude = 1)
    phase_mask_complex = tf.exp(1j * tf.cast(phase_mask, tf.complex64))
    # Propagate through ASM to get simulated intensity (hotplot)
    holo_prop = tf.abs(asm_gpu(phase_mask_complex, x1, lamb, size_padded, size_padded, pixel_size)) ** 2
    # Loss compares the propagated intensity with the target intensity (for one slice)
    loss = tf.reduce_mean(tf.square(holo_prop - x2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Choose a propagation distance z (here 5mm)
    z_val = np.array([5 * mm], dtype=np.complex64)

    # Iterative optimization loop
    for epoch in range(train_epochs):
        _, cur_loss = sess.run([optimizer, loss], feed_dict={
            x1: z_val,
            x2: holo_target[:, :, 0]  # using the first slice as the target intensity
        })
        print(f"[Iterative] Epoch {epoch+1}/{train_epochs}, loss = {cur_loss:.6f}")

    # Retrieve the optimized phase mask (values in [0, 2π])
    final_phase = sess.run(phase_mask)
    sess.close()
    # Crop out the hologram region by removing padding (resulting in a 2D array)
    final_phase_cropped = final_phase[size_pad:size_pad+size_holo, size_pad:size_pad+size_holo]
    # Normalize the phase mask to [0,1] by dividing by 2π
    final_phase_norm = final_phase_cropped / (2 * np.pi)
    print(f"[Iterative] Finished processing {input_img_path}")
    return final_phase_norm

# ========== END OF ITERATIVE CODE ==========

# -------------------------------
# PART B: DATASET & TRAINING IN PYTORCH (with caching & multi-channel input)
# -------------------------------
class HotplotDataset(Dataset):
    """
    A PyTorch Dataset that:
      1) Reads each ideal target image from a folder.
      2) Checks if a cached phase mask (.npy file) exists; if not, computes it.
      3) Creates a multi-channel input by replicating the image over num_layers_z channels.
      4) Returns (input_intensity_stack, phase_mask) as tensors.
         - Input: shape (num_layers_z, H, W)
         - Target: shape (1, H, W), the optimized phase mask (normalized to [0,1])
    """
    def __init__(self, img_dir, cache_dir="cache_hotplots", num_layers_z=4, augment=True):
        super().__init__()
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.num_layers_z = num_layers_z
        self.augment = augment
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"[Cache] Created cache directory at {self.cache_dir}")
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        print(f"[Dataset] Found {len(self.img_files)} images in {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        cache_file = os.path.join(self.cache_dir, img_name + "_phase.npy")

        # Load the ideal intensity target image
        img_pil = Image.open(img_path).convert('L')
        img_np = np.array(img_pil, dtype=np.float32) / 255.0

        # Create multi-channel input by replicating over num_layers_z channels
        input_stack = np.stack([img_np] * self.num_layers_z, axis=-1)  # shape (H, W, num_layers_z)
        input_stack = np.transpose(input_stack, (2, 0, 1))  # rearrange to (num_layers_z, H, W)

        # Check if phase mask is cached; if not, compute it using iterative optimization.
        if os.path.exists(cache_file):
            print(f"[Cache] Loading cached phase mask for {img_name}")
            phase_mask_np = np.load(cache_file)
        else:
            print(f"[Cache] Computing phase mask for {img_name}")
            phase_mask_np = compute_phase_mask_iterative(img_path, train_epochs=3)
            print(f"[Cache] Saving phase mask for {img_name} to {cache_file}")
            np.save(cache_file, phase_mask_np)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(input_stack).float()      # shape: (num_layers_z, H, W)
        # The target is now the phase mask (a single 2D array normalized to [0,1]) with an added channel dimension.
        y_tensor = torch.from_numpy(phase_mask_np.astype(np.float32)).unsqueeze(0)  # shape: (1, H, W)

        # Optionally apply data augmentation (random flips)
        if self.augment:
            x_tensor, y_tensor = self.random_flip(x_tensor, y_tensor)

        return x_tensor, y_tensor

    def random_flip(self, x, y):
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        return x, y

# A simple U-Net that takes multi-channel input and outputs a single-channel phase mask.
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, features=32):
        super(SimpleUNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.up(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# -------------------------------
# MAIN FUNCTION: TRAIN, TEST, VISUALIZE
# -------------------------------
def main():
    # 1) CREATE DATASET
    input_dir = r"datasets\50pngs"  # Folder with ~50 training images; adjust path as needed.
    num_layers_z = 4  # Number of slices (channels) for the input stack
    dataset = HotplotDataset(input_dir, cache_dir="cache_hotplots", num_layers_z=num_layers_z, augment=True)
    print(f"[Main] Found {len(dataset)} images.")

    # Train-test split (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"[Main] Training on {train_size} images, testing on {test_size} images.")

    # Dataloaders
    batch_size = 1  # Small batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2) CREATE MODEL & OPTIMIZER
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet(in_channels=num_layers_z, out_channels=1, features=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3) TRAIN
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"[Train] Starting epoch {epoch+1}/{num_epochs}...")
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"[Train] Epoch {epoch+1}, sample {i+1}/{len(train_loader)}: Loss = {loss.item():.6f}")
        avg_loss = running_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1} finished with average loss: {avg_loss:.6f}")

    # 4) TEST
    model.eval()
    test_loss = 0.0
    print("[Test] Starting evaluation...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            print(f"[Test] Sample {i+1}/{len(test_loader)}: Loss = {loss.item():.6f}")
    test_loss /= len(test_loader)
    print(f"[Test] Final Test MSE Loss: {test_loss:.6f}")

    # 5) VISUALIZE PREDICTIONS
    def display_predictions(model, test_loader, device, num_samples=5):
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                # x shape: (B, C, H, W) => average over channels for display
                input_img = x.cpu().squeeze(0).mean(dim=0).numpy()  # shape (H, W)
                # y and y_pred: shape (B, 1, H, W) => squeeze batch and channel dims
                target_phase = y.cpu().squeeze(0).squeeze(0).numpy()  # shape (H, W)
                predicted_phase = y_pred.cpu().squeeze(0).squeeze(0).numpy()  # shape (H, W)
                
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(input_img, cmap='gray')
                axs[0].set_title('Input (Avg Intensity)')
                axs[1].imshow(target_phase, cmap='gray')
                axs[1].set_title('Target Phase Mask (norm)')
                axs[2].imshow(predicted_phase, cmap='gray')
                axs[2].set_title('Predicted Phase Mask (norm)')
                for ax in axs:
                    ax.axis('off')
                plt.show()
                
                if i+1 >= num_samples:
                    break

    display_predictions(model, test_loader, device, num_samples=5)

if __name__ == "__main__":
    main()