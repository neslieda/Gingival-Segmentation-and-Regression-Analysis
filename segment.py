import json
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os

coco_json_path = r"C:\Users\edayu\PycharmProjects\Yapayzeka\dental\dental\dental\gazi_gum_project\formatted_file.json"  # COCO JSON dosyanızın yolu
output_csv_path = r"C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\measurements.csv"  # Çıkış CSV dosya yolu
output_masks_dir = r"C:\Users\edayu\PycharmProjects\Yapayzeka\dental\Diş eti akademik yapay zeka\masks"  # Maskların kaydedileceği dizin
# Sonuçları saklamak için bir DataFrame oluşturuyoruz
measurements_df = pd.DataFrame(columns=['image_id', 'file_name', 'measurement_type', 'x', 'y'])

with open(coco_json_path, 'r') as json_file:
    coco_data = json.load(json_file)

for item in coco_data['images']:
    target_image_data = item
    target_image_id = target_image_data['id']
    if target_image_data:
        ## mask create ##

        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == target_image_id]
        mask = Image.new('L', (target_image_data['width'], target_image_data['height']), 0)
        draw = ImageDraw.Draw(mask)

        for annotation in annotations:
            segmentation = annotation['segmentation']
            category_id = annotation['category_id']

            for segment in segmentation:
                draw.polygon(segment, outline=None,
                             fill=255)  # fill parametresini sabit bir değerle değiştirin (255 beyaz renktir)

        ## middle point ##

        numpy_array = np.array(mask)
        non_black_pixels_mask = numpy_array > 0
        height, width = non_black_pixels_mask.shape
        start = int(width * 0.47)
        end = int(width * 0.53)
        points_of_middle = []
        for item in range(start, end + 1):
            res_tmp = np.unique(non_black_pixels_mask[:, item], return_counts=True)
            if res_tmp[0][0] == True:
                points_of_middle.append(res_tmp[1][0])
            else:
                points_of_middle.append(res_tmp[1][1])

        middle_point = points_of_middle.index(max(points_of_middle)) + start
        image_draw = ImageDraw.Draw(mask)
        draw.line((middle_point, 0, middle_point, height), fill="black", width=1)

        # Orta noktayı DataFrame'e ekleme
        measurements_df = pd.concat([measurements_df, pd.DataFrame([{
            'image_id': target_image_id,
            'file_name': target_image_data['file_name'],
            'measurement_type': 'middle',
            'x': middle_point,
            'y': 0
        }])], ignore_index=True)

        ## draw poi ##

        ## right ##
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
            local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].loc[(df != 0).any(axis=1)]
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
            measurements_df = pd.concat([measurements_df, pd.DataFrame([{
                'image_id': target_image_id,
                'file_name': target_image_data['file_name'],
                'measurement_type': 'right',
                'x': points,
                'y': 0
            }])], ignore_index=True)

        ## left ##
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
            local_min_vals = df.loc[df['data'] == df['data'].rolling(n, center=True).min()].loc[(df != 0).any(axis=1)]
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
            measurements_df = pd.concat([measurements_df, pd.DataFrame([{
                'image_id': target_image_id,
                'file_name': target_image_data['file_name'],
                'measurement_type': 'left',
                'x': points,
                'y': 0
            }])], ignore_index=True)

        # Mask dosyasını kaydet
        mask_file_path = os.path.join(output_masks_dir, f"{target_image_data['file_name']}_mask.png")
        mask.save(mask_file_path)

    else:
        print(f"Belirtilen image_id'ye sahip görsel bulunamadı: {target_image_id}")

# CSV dosyasını kaydet
measurements_df.to_csv(output_csv_path, index=False)
