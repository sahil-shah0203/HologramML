"""
ONE CODE BLOCK:
1) Iterative hologram-optimization code (TensorFlow) 
2) Loop over ~50 training images -> Generate and cache hotplots
3) Simple PyTorch model to train on (image -> hotplot) pairs
4) Test for accuracy

Adjust folder paths, parameters, batch sizes, etc. as needed.
"""

# -------------------------------
# PART A: IMPORTS & ITERATIVE CODE
# -------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -- TensorFlow (for iterative code)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# -- PyTorch (for model training)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ========== ITERATIVE CODE STARTS HERE ==========
def compute_hotplot_iterative(input_img_path: str,
                              size_holo=1600,
                              size_pad=300,
                              num_layers_z=4,
                              num_stop=8,
                              learning_rate=0.003,
                              train_epochs=10):
    """
    Adapted from your iterative code.
    This function:
      1) Loads one input image from input_img_path.
      2) Resizes it to (size_padded, size_padded).
      3) Creates a 'holo_target' stack by replicating the image.
      4) Runs TF optimization to find a phase mask.
      5) Returns the final propagated intensity (hotplot) from one plane.
    """
    print(f"[Iterative] Processing image: {input_img_path}")
    # -------------- Setup constants --------------
    nm = 1e-9; um = 1e-6; mm = 1e-3
    lamb = 405 * nm   # Operating wavelength
    pixel_size = 1.5 * um
    size_padded = size_holo + 2 * size_pad

    # -------------- Load & resize input image --------------
    img_pil = Image.open(input_img_path).convert('L')
    img_pil = img_pil.resize((size_padded, size_padded), Image.BICUBIC)
    img_np = np.array(img_pil, dtype=np.float32) / 255.0

    # Stack the same image across num_layers_z planes (as placeholder)
    holo_target = np.stack([img_np] * num_layers_z, axis=-1)

    # -------------- ASM Prep --------------
    dfx = 1 / size_padded / pixel_size
    dfy = 1 / size_padded / pixel_size
    k_x = dfx * (np.linspace(-size_padded/2, size_padded/2 - 1, size_padded))
    k_y = dfy * (np.linspace(-size_padded/2, size_padded/2 - 1, size_padded))
    fx, fy = np.meshgrid(k_x, -k_y)
    p = (fx**2 + fy**2) * (lamb**2)
    p = p.astype(np.float32)

    # define ASM in TF
    def asm_gpu(ob, z, lam, mx, my, pixel_size):
        p_tf = tf.constant(p)
        sp = tf.sqrt(1 - p_tf)
        sp = tf.cast(sp, tf.complex64)
        factor = 2 * np.pi * 1j * z / lam
        q = tf.signal.fftshift(tf.exp(factor * sp))
        return tf.ifft2d(tf.fft2d(ob) * q)

    # -------------- Build TF graph --------------
    x1 = tf.placeholder(tf.complex64, [1])   # z-plane
    x2 = tf.placeholder(tf.float32, [size_padded, size_padded])  # target intensity
    phase1 = tf.random_normal([size_padded, size_padded])
    phase_mask = tf.Variable(phase1, dtype=tf.float32)
    phase_mask = 2 * np.pi * (tf.mod(phase_mask, 1))
    phase_mask_complex = tf.exp(1j * tf.cast(phase_mask, tf.complex64))
    holo_prop = tf.abs(asm_gpu(phase_mask_complex, x1, lamb, size_padded, size_padded, pixel_size)) ** 2
    loss = tf.reduce_mean(tf.square(holo_prop - x2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # -------------- Run optimization --------------
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    z_val = np.array([5 * mm], dtype=np.complex64)

    for epoch in range(train_epochs):
        _, cur_loss = sess.run([optimizer, loss], feed_dict={
            x1: z_val,
            x2: holo_target[:, :, 0]  # using plane 0 as target
        })
        print(f"[Iterative] Epoch {epoch+1}/{train_epochs}, loss = {cur_loss:.6f}")

    # -------------- Get final hotplot --------------
    final_phase = sess.run(phase_mask)
    final_phase_complex = np.exp(1j * final_phase)

    def asm_cpu(ob, z, lam, mx, my, pixel_size):
        dfx_ = 1 / mx / pixel_size
        dfy_ = 1 / my / pixel_size
        kx_ = dfx_ * (np.linspace(-mx/2, mx/2 - 1, mx))
        ky_ = dfy_ * (np.linspace(-my/2, my/2 - 1, my))
        fx_, fy_ = np.meshgrid(kx_, -ky_)
        p_ = (fx_**2 + fy_**2) * (lam**2)
        sp_ = np.sqrt(1 - p_)
        q_ = np.fft.fftshift(np.exp(2 * np.pi * 1j * z / lam * sp_))
        field = np.fft.ifft2(np.fft.fft2(ob) * q_)
        return field

    field_out = asm_cpu(final_phase_complex, 5 * mm, lamb, size_padded, size_padded, pixel_size)
    hotplot_out = np.abs(field_out) ** 2

    sess.close()
    print(f"[Iterative] Finished processing {input_img_path}")
    return hotplot_out

# ========== END OF ITERATIVE CODE ==========

# -------------------------------
# PART B: DATASET & TRAINING IN PYTORCH (with caching)
# -------------------------------
class HotplotDataset(Dataset):
    """
    A PyTorch Dataset that:
      1) Reads each input image from a folder.
      2) Checks if a cached hotplot (.npy file) exists; if not, computes it.
      3) Returns (input_image, hotplot) as tensors.
    """
    def __init__(self, img_dir, cache_dir="cache_hotplots"):
        super().__init__()
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"[Cache] Created cache directory at {self.cache_dir}")
        self.img_files = sorted([f for f in os.listdir(img_dir)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        print(f"[Dataset] Found {len(self.img_files)} images in {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        cache_file = os.path.join(self.cache_dir, img_name + "_hotplot.npy")

        # Load the input image
        img_pil = Image.open(img_path).convert('L')
        img_np = np.array(img_pil, dtype=np.float32) / 255.0

        # Check if hotplot is cached
        if os.path.exists(cache_file):
            print(f"[Cache] Loading cached hotplot for {img_name}")
            hotplot_np = np.load(cache_file)
        else:
            print(f"[Cache] Computing hotplot for {img_name}")
            hotplot_np = compute_hotplot_iterative(img_path, train_epochs=3)
            hotplot_np = hotplot_np / (np.max(hotplot_np) + 1e-8)  # Normalize
            print(f"[Cache] Saving hotplot for {img_name} to {cache_file}")
            np.save(cache_file, hotplot_np)

        # Convert to PyTorch tensors with shape (1, H, W) and type float32
        x_tensor = torch.from_numpy(img_np.astype(np.float32)).unsqueeze(0)
        y_tensor = torch.from_numpy(hotplot_np.astype(np.float32)).unsqueeze(0)
        return x_tensor, y_tensor

# A small U-Net style network
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(SimpleUNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(features, features, 3, padding=1)
        self.conv4 = nn.Conv2d(features, out_channels, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.up(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

def main():
    # -------------------------------
    # 1) CREATE DATASET
    # -------------------------------
    input_dir = r"datasets\50pngs"  # Folder with ~50 training images (adjust this path)
    dataset = HotplotDataset(input_dir)
    print(f"[Main] Found {len(dataset)} images.")

    # Train-test split (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"[Main] Training on {train_size} images, testing on {test_size} images.")

    # Dataloaders
    batch_size = 1  # Use a small batch size due to heavy computation per image
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------
    # 2) CREATE MODEL & OPTIMIZER
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------------
    # 3) TRAIN
    # -------------------------------
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

    # -------------------------------
    # 4) TEST
    # -------------------------------
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


    # -------------------------------
    # VISUALIZE PREDICTIONS
    # -------------------------------
    import matplotlib.pyplot as plt

    def display_predictions(model, test_loader, device, num_samples=5):
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                
                # Convert the tensors to numpy arrays (remove batch and channel dimensions)
                input_img = x.cpu().squeeze().numpy()
                target_hotplot = y.cpu().squeeze().numpy()
                predicted_hotplot = y_pred.cpu().squeeze().numpy()
                
                # Plot the input, target, and prediction side-by-side
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(input_img, cmap='gray')
                axs[0].set_title('Input Image')
                axs[1].imshow(target_hotplot, cmap='gray')
                axs[1].set_title('Target Hotplot')
                axs[2].imshow(predicted_hotplot, cmap='gray')
                axs[2].set_title('Predicted Hotplot')
                for ax in axs:
                    ax.axis('off')
                plt.show()
                
                if i+1 >= num_samples:
                    break

    # Call the function to display predictions for a few test samples.
    display_predictions(model, test_loader, device, num_samples=5)


if __name__ == "__main__":
    main()
