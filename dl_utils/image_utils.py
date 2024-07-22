import numpy as np

def make_grid(images):
    """
    :param images: numpy.ndarray
        shape (N, C, H, W)
    :return: numpy.ndarray
        shape (H, W, C) or (H, W)
    """
    num_images, channels, height, width = images.shape

    # If channels are 1 or 3, create a normal grid
    if channels == 1 or channels == 3:
        # Calculate number of rows and columns for the grid
        nrow = int(np.sqrt(num_images))
        ncol = np.ceil(num_images / nrow).astype(int)

        # Initialize an empty array for the grid
        grid_height = nrow * height
        grid_width = ncol * width
        grid = np.zeros((grid_height, grid_width, channels), dtype=images.dtype)

        # Populate the grid
        for i in range(num_images):
            row = i // ncol
            col = i % ncol
            grid[row*height:(row+1)*height, col*width:(col+1)*width, :] = images[i].transpose(1, 2, 0)

        if channels == 1:
            grid = grid.squeeze(-1)  # Remove the last dimension for grayscale images

        return grid

    # Else it is a 3D image where we have to concatenate the slices along the x-axis
    else:
        concatenated_slices = []

        for i in range(num_images):
            # Extract slices from the 3D image
            slices = [images[i, j] for j in range(channels)]

            # Concatenate slices horizontally
            concatenated = np.concatenate(slices, axis=1)
            concatenated_slices.append(concatenated)

        # Stack the concatenated slices vertically
        stacked_grid = np.concatenate(concatenated_slices, axis=0)

        return stacked_grid

def make_comparison_grid(images1, images2):
    """
    :param images1: numpy.ndarray
        shape (N, C, H, W)
    :param images2: numpy.ndarray
        shape (N, C, H, W)
    :return: numpy.ndarray
        shape (H, W, C) or (H, W)
    """
    # Stack images vertically
    images = np.concatenate([images1, images2], axis=2)  # Concatenate along the height
    return make_grid(images)

def img_3d_to_2d(img: np.ndarray):
    """
    :param img: numpy.ndarray
        shape (C, H, W)
    :return: numpy.ndarray
        shape (1, H*a, W*b)
    """
    if len(img.shape) != 3:
        raise ValueError(f'Expected 3D image, got image with shape {img.shape}')

    if img.shape[0] == 1:
        return img

    # Concatenate slices (0th dim) horizontally
    img = [img[i] for i in range(img.shape[0])]
    img = np.concatenate(img, axis=1)

    return img


