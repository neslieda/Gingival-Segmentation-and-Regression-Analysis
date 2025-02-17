import os
import json
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

# Paths to the folders
json_folder_path = r"C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\labels"
image_folder_path = r"C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\images"
output_folder_path = r"C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\gum2\gum\masks"

# Create output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Iterate through each JSON file in the folder
json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

for json_file_name in json_files:
    json_file_path = os.path.join(json_folder_path, json_file_name)

    with open(json_file_path, 'r') as json_file:
        coco_data = json.load(json_file)

    for item in coco_data['images']:
        target_image_data = item
        target_image_id = target_image_data['id']
        image_file_name = target_image_data['file_name']

        if target_image_data:
            # Create mask
            annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == target_image_id]
            mask = Image.new('L', (target_image_data['width'], target_image_data['height']), 0)
            draw = ImageDraw.Draw(mask)

            for annotation in annotations:
                segmentation = annotation['segmentation']
                for segment in segmentation:
                    draw.polygon(segment, outline=None, fill=255)

            # Determine middle point
            numpy_array = np.array(mask)
            non_black_pixels_mask = numpy_array > 0
            height, width = non_black_pixels_mask.shape
            start = int(width * 0.47)
            end = int(width * 0.53)
            points_of_middle = []
            for item in range(start, end + 1):
                res_tmp = np.unique(non_black_pixels_mask[:, item], return_counts=True)
                if len(res_tmp[0]) > 1:
                    points_of_middle.append(res_tmp[1][1])
                else:
                    points_of_middle.append(res_tmp[1][0])
            middle_point = points_of_middle.index(max(points_of_middle)) + start

            # Draw line at middle point
            draw.line((middle_point, 0, middle_point, height), fill="black", width=1)

            # Determine points of interest on the right
            n = 1
            test = []
            before = 0
            before_count = 0
            for item in range(middle_point, non_black_pixels_mask.shape[1]):
                white_count = np.sum(non_black_pixels_mask[:, item])
                if before == white_count and white_count > 0:
                    before_count = before_count + 1
                    white_count_app = white_count - (0.001 * before_count)
                else:
                    before_count = 0
                    white_count_app = white_count
                before = white_count
                test.append(white_count_app)
            df = pd.DataFrame(test, columns=['data'])  # rolling period
            while True:
                local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].loc[
                    (df != 0).any(axis=1)]
                bef_index = -1000
                bef_item = -1000
                dropped_index = []
                for index, item in local_min_vals.iterrows():
                    if index - bef_index <= 30:
                        if bef_item['data'] > item['data']:
                            dropped_index.append(bef_index)
                        else:
                            dropped_index.append(index)
                    bef_index = index
                    bef_item = item
                local_min_vals.drop(index=dropped_index, inplace=True)
                n = n + 1
                if len(local_min_vals) <= 3:
                    break

            for index, item in local_min_vals.iterrows():
                points = middle_point + index
                draw.line((points, 0, points, height), fill="black", width=1)

            # Determine points of interest on the left
            n = 1
            test = []
            before = 0
            before_count = 0
            for item in range(middle_point, 0, -1):
                white_count = np.sum(non_black_pixels_mask[:, item])
                if before == white_count and white_count > 0:
                    before_count = before_count + 1
                    white_count_app = white_count - (0.001 * before_count)
                else:
                    before_count = 0
                    white_count_app = white_count
                before = white_count
                test.append(white_count_app)
            df = pd.DataFrame(test, columns=['data'])  # rolling period
            while True:
                local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].loc[
                    (df != 0).any(axis=1)]
                bef_index = -1000
                bef_item = -1000
                dropped_index = []
                for index, item in local_min_vals.iterrows():
                    if index - bef_index <= 30:
                        if bef_item['data'] > item['data']:
                            dropped_index.append(bef_index)
                        else:
                            dropped_index.append(index)
                    bef_index = index
                    bef_item = item
                local_min_vals.drop(index=dropped_index, inplace=True)
                n = n + 1
                if len(local_min_vals) <= 3:
                    break

            for index, item in local_min_vals.iterrows():
                points = middle_point - index
                draw.line((points, 0, points, height), fill="black", width=1)

            # Save the mask
            mask.save(os.path.join(output_folder_path, f"mask_{image_file_name}"))
        else:
            print(f"Belirtilen image_id'ye sahip görsel bulunamadı: {target_image_id}")
