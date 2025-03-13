import os
import json

# Path to the main folder containing subfolders with result.json files
main_folder_path = '../mnt/data/data_for_paper/result/ours_3w'

# Initialize counters and sums for each metric
ssim_total = 0
psnr_total = 0
lpips_total = 0
file_count = 0

# Traverse through each subfolder and read the result.json file
for root, dirs, files in os.walk(main_folder_path):
    for file in files:
        if file == 'results.json':
            file_count += 1
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract metrics from 'ours_50000' key
                ssim_total += data["ours_50000"]["SSIM"]
                psnr_total += data["ours_50000"]["PSNR"]
                lpips_total += data["ours_50000"]["LPIPS"]

# Calculate averages if at least one file was found
if file_count > 0:
    ssim_avg = ssim_total / file_count
    psnr_avg = psnr_total / file_count
    lpips_avg = lpips_total / file_count

    # Print the average metrics
    print(f"Average SSIM: {ssim_avg}")
    print(f"Average PSNR: {psnr_avg}")
    print(f"Average LPIPS: {lpips_avg}")
else:
    print("No result.json files found.")

