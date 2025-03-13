import os
import cv2
import numpy as np


# Function to compute and save error maps
def generate_error_maps(ground_truth_folder, rendered_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of images in ground truth folder
    ground_truth_images = sorted(os.listdir(ground_truth_folder))
    rendered_images = sorted(os.listdir(rendered_folder))

    # Check that the folders have the same number of images
    if len(ground_truth_images) != len(rendered_images):
        print("The number of images in the folders does not match.")
        return

    # Process each image pair
    for img_name in ground_truth_images:
        ground_truth_path = os.path.join(ground_truth_folder, img_name)
        rendered_path = os.path.join(rendered_folder, img_name)

        # Load the images
        ground_truth_img = cv2.imread(ground_truth_path)
        rendered_img = cv2.imread(rendered_path)

        # Check if the images were loaded properly
        if ground_truth_img is None or rendered_img is None:
            print(f"Error loading images: {img_name}")
            continue

        # Check if images have the same size
        if ground_truth_img.shape != rendered_img.shape:
            print(f"Size mismatch for image: {img_name}")
            continue

        # Compute absolute difference (error map)
        error_map = cv2.absdiff(ground_truth_img, rendered_img)

        # Convert the error map to grayscale if necessary (single-channel)
        error_map_gray = cv2.cvtColor(error_map, cv2.COLOR_BGR2GRAY)

        # Normalize the grayscale error map to the range [0, 255]
        norm_error_map = np.zeros_like(error_map_gray)
        cv2.normalize(error_map_gray, norm_error_map, 0, 255, cv2.NORM_MINMAX)

        # Apply a red/blue color map (COLORMAP_JET) to the normalized error map
        colored_error_map = cv2.applyColorMap(norm_error_map, cv2.COLORMAP_VIRIDIS)

        # Save the colorized error map to the output folder
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, colored_error_map)

        print(f"Error map saved for: {img_name}")


# Define the folder paths
# ground_truth_folder = '../mnt/data/ours_result_GS/meeting1018/teaser_img/train_10082024_xyzw_exp_initdist_wlr0.0002_end1e-6_5witr/test/ours_50000/gt'
# rendered_folder = '../mnt/data/ours_result_GS/meeting1018/teaser_img/train_10082024_xyzw_exp_initdist_wlr0.0002_end1e-6_5witr/test/ours_50000/renders'
# output_folder = '../mnt/data/ours_result_GS/meeting1018/teaser_img/train_10082024_xyzw_exp_initdist_wlr0.0002_end1e-6_5witr/error_map_test_VIRIDIS'

# ground_truth_folder = '../mnt/data/ours_result_GS/meeting1018/test_nerfbaselines_INGP_train/output/predictions/test/INGP/gt'
# rendered_folder = '../mnt/data/ours_result_GS/meeting1018/test_nerfbaselines_INGP_train/output/predictions/test/INGP/renders'
# output_folder = '../mnt/data/ours_result_GS/meeting1018/test_nerfbaselines_INGP_train/output/predictions/error_map_test_VIRIDIS'

ground_truth_folder = '../mnt/data/data_for_paper/result/train_ours_5w_11112024/test/ours_50000/gt'
rendered_folder = '../mnt/data/data_for_paper/result/train_ours_5w_11112024/test/ours_50000/renders'
output_folder = '../mnt/data/data_for_paper/result/train_ours_5w_11112024/error_map_test_VIRIDIS'

# Generate error maps
generate_error_maps(ground_truth_folder, rendered_folder, output_folder)
