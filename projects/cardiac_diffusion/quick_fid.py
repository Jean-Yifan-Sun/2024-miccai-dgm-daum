import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from scipy.linalg import sqrtm
import numpy as np

import SimpleITK as sitk
from PIL import Image
import numpy as np

# Function to extract features using Inception v3
def get_features(images, model, device):
    preprocess = Compose([
        Resize((299, 299)),  # Inception expects 299x299 images
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        Normalize(mean=[0.485],std=[0.229])
    ])
    images = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        features = model(images).detach().cpu().numpy()
    return features

# FID computation
def calculate_fid(real_features, fake_features):
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    # Compute squared difference of means
    mean_diff = np.sum((mu_real - mu_fake) ** 2)
    
    # Compute sqrt of product of covariance matrices
    cov_mean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(cov_mean):  # Handle numerical instability
        cov_mean = cov_mean.real

    fid = mean_diff + np.trace(sigma_real + sigma_fake - 2 * cov_mean)
    return fid

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(device)
inception_model.fc = nn.Identity()  # Use the penultimate layer for features

real_images = '/rds/projects/c/chenhp-dpmodel/2024-miccai-dgm-daum/projects/cardiac_diffusion/weights/cardiac_diffusion/2024_11_03_20_32_37_584402/frd_images/real_images/101577771508687736.nii.gz'
generated_images = '/rds/projects/c/chenhp-dpmodel/2024-miccai-dgm-daum/projects/cardiac_diffusion/weights/cardiac_diffusion/2024_11_03_20_32_37_584402/frd_images/fake_images/101577771508687736.nii.gz'


def to_pil(path):
    # Load the .nii.gz file
    nii_path = path
    image = sitk.ReadImage(nii_path)

    # Convert the SimpleITK image to a numpy array
    image_array = sitk.GetArrayFromImage(image)  # Shape: (depth, height, width)
    print(image.GetSize())
    # Process each 2D slice (depth dimension)
    slices = []
    for i, slice_2d in enumerate(image_array):
        # Normalize the image slice to range [0, 255] for visualization
        slice_2d_normalized = ((slice_2d - slice_2d.min()) / (slice_2d.ptp()) * 255).astype(np.uint8)

        # Convert to a PIL Image
        pil_image = Image.fromarray(slice_2d_normalized)

        # Save or display the image
        slices.append(pil_image)
        # pil_image.save(f"slice_{i}.png")  # Save as PNG
        # pil_image.show()  # Show the image
        return slices

real_images = to_pil(real_images)
generated_images = to_pil(generated_images)

# Assume `real_images` and `generated_images` are lists of PIL images
real_features = get_features(real_images, inception_model, device)
fake_features = get_features(generated_images, inception_model, device)

fid_score = calculate_fid(real_features, fake_features)
print(f"FID Score: {fid_score}")
