import json
from PIL import Image, ImageDraw
import os

# COCO JSON dosyasını oku
coco_json_path = r'C:\Users\edayu\PycharmProjects\Dental\DentalProject\dental\dental\gazi_gum_project\formatted_file.json'

with open(coco_json_path, 'r') as json_file:
    coco_data = json.load(json_file)

# İlgili görsel için bilgileri seçin
target_image_id = 17

# İlgili görselin COCO verilerini bulun
target_image_data = next((img for img in coco_data['images'] if img['id'] == target_image_id), None)
if target_image_data:
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == target_image_id]

    # Maskeyi oluşturun
    mask = Image.new('L', (target_image_data['width'], target_image_data['height']), 0)
    draw = ImageDraw.Draw(mask)

    for annotation in annotations:
        segmentation = annotation['segmentation']
        category_id = annotation['category_id']
        for segment in segmentation:
            draw.polygon(segment, outline=None, fill=255)

    # Klasörün varlığını kontrol et ve yoksa oluştur
    masks_folder_path = '/local/masks/'
    if not os.path.exists(masks_folder_path):
        os.makedirs(masks_folder_path)

    # Maskeyi BMP olarak kaydedin
    mask_save_path = os.path.join(masks_folder_path, f'{target_image_data["file_name"]}_mask.bmp')
    mask.save(mask_save_path)

    print(f"Maske kaydedildi: {mask_save_path}")
else:
    print(f"Belirtilen image_id'ye sahip görsel bulunamadı: {target_image_id}")

import numpy as np

# Maskeyi oku
img = Image.open(mask_save_path)
numpy_array = np.array(img)
non_black_pixels_mask = numpy_array > 0

height, width = non_black_pixels_mask.shape

# Orta noktayı bul
start = int(width * 0.4)
end = int(width * 0.6)

points_of_middle = []
for item in range(start, end + 1):
    res_tmp = np.unique(non_black_pixels_mask[:, item], return_counts=True)
    if res_tmp[0][0] == True:
        points_of_middle.append(res_tmp[1][0])
    else:
        points_of_middle.append(res_tmp[1][1])

middle_point = points_of_middle.index(max(points_of_middle)) + start

# Tepe noktalarını bulma fonksiyonu
def find_peaks(non_black_pixels_mask, middle_point, width):
    total_count = 0
    break_parameter = 6
    points_of_interest = []

    while total_count != 3:
        before = 0
        cont_increase = 0
        cont_decrease = 0
        count = []
        index_number = 0
        points_of_interest = []

        for item in range(middle_point, width):
            if len(points_of_interest) == 3:
                break
            white_count = len(non_black_pixels_mask[:, item][np.where(non_black_pixels_mask[:, item] == True)])
            if middle_point == item:
                before = white_count
            else:
                if cont_increase < break_parameter:
                    if white_count < before:
                        before = white_count
                        cont_increase += 1
                else:
                    if cont_decrease == break_parameter:
                        if len(count) > 0:
                            points_of_interest.append(count.index(min(count)) + index_number)
                        count = []
                        cont_increase = 0
                        cont_decrease = 0
                    else:
                        if cont_increase == break_parameter:
                            count.append(white_count)
                            if len(count) == 1:
                                index_number = item
                            if white_count > before:
                                before = white_count
                                cont_decrease += 1
                            else:
                                before = white_count
                                cont_decrease = 0
        break_parameter -= 1
        total_count = len(points_of_interest)
        if break_parameter == 0:
            break

    return points_of_interest

right_peaks = find_peaks(non_black_pixels_mask, middle_point, width)
left_peaks = find_peaks(non_black_pixels_mask[:, ::-1], middle_point, width)
left_peaks = [(width - col) for col in left_peaks]

all_peaks = sorted(right_peaks + left_peaks)

# En yüksek noktayı bulmak
highest_peak = min(all_peaks, key=lambda x: np.min(np.where(non_black_pixels_mask[:, x])))

print(f"En yüksek nokta: {highest_peak}")

def draw_lines_on_image(draw, points_of_interest, height):
    for point in points_of_interest:
        draw.line((point, 0, point, height), fill="black", width=1)
        top_pixel = np.min(np.where(non_black_pixels_mask[:, point]))
        draw.line((point, top_pixel, point, top_pixel + 10), fill="red", width=2)  # En tepe noktayı vurgulama

def draw_lines_and_save(img_path, output_path):
    img = Image.open(img_path)
    numpy_array = np.array(img)
    non_black_pixels_mask = numpy_array > 0

    height, width = non_black_pixels_mask.shape
    middle_point = int(width * 0.5)

    right_peaks = find_peaks(non_black_pixels_mask, middle_point, width)
    left_peaks = find_peaks(non_black_pixels_mask[:, ::-1], middle_point, width)
    left_peaks = [(width - col) for col in left_peaks]

    all_peaks = sorted(right_peaks + left_peaks)
    highest_peak = min(all_peaks, key=lambda x: np.min(np.where(non_black_pixels_mask[:, x])))

    img_with_lines = img.copy()
    draw = ImageDraw.Draw(img_with_lines)

    draw_lines_on_image(draw, all_peaks, height)

    img_with_lines.save(output_path)

if __name__ == "__main__":
    img_path = "/local/masks/2024-02-20 230833.png_mask.bmp"
    output_path = "/local/masked_image_with_lines5.bmp"
    draw_lines_and_save(img_path, output_path)
