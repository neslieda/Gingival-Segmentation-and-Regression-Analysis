import json
from PIL import Image, ImageDraw
import numpy as np
import os


# Function to load COCO JSON data
def load_coco_json(coco_json_path):
    with open(coco_json_path, 'r') as json_file:
        coco_data = json.load(json_file)
    return coco_data


# Function to create mask from annotations
def create_mask(target_image_data, annotations):
    mask = Image.new('L', (target_image_data['width'], target_image_data['height']), 0)
    draw = ImageDraw.Draw(mask)

    for annotation in annotations:
        for segment in annotation['segmentation']:
            draw.polygon(segment, outline=None, fill=255)
    return mask


# Function to save the mask image
def save_mask(mask, file_name):
    masks_folder_path = 'local/masks/'
    if not os.path.exists(masks_folder_path):
        os.makedirs(masks_folder_path)

    mask_save_path = os.path.join(masks_folder_path, f'{file_name}_mask.bmp')
    mask.save(mask_save_path)
    return mask_save_path


# Function to count white pixels in a given column
def count_white_pixels_in_column(non_black_pixels_mask, column):
    return np.sum(non_black_pixels_mask[:, column])


# Improved function to find the middle point
def find_middle_point(non_black_pixels_mask, start_percentage=0.4, end_percentage=0.6):
    width = non_black_pixels_mask.shape[1]
    start = int(width * start_percentage)
    end = int(width * end_percentage)

    points_of_middle = []
    for item in range(start, end + 1):
        white_count = count_white_pixels_in_column(non_black_pixels_mask, item)
        points_of_middle.append((item, white_count))

    # Find the column with the maximum white pixels
    max_white_count = max(points_of_middle, key=lambda x: x[1])[1]
    # Filter columns with values close to the maximum
    close_points = [p for p in points_of_middle if abs(p[1] - max_white_count) < max_white_count * 0.1]
    # Calculate the average column index of these points
    middle_point = int(np.mean([p[0] for p in close_points]))

    return middle_point


# Function to find interesting points based on white pixel counts
def find_interesting_points(non_black_pixels_mask, middle_point, break_parameter=6):
    total_count = 0
    points_of_interest = []

    while total_count != 3:
        before = 0
        cont_increase = 0
        cont_decrease = 0
        count = []
        index_number = 0
        points_of_interest = []

        for item in range(middle_point, non_black_pixels_mask.shape[1]):
            if len(points_of_interest) == 3:
                break
            white_count = np.sum(non_black_pixels_mask[:, item])
            if middle_point == item:
                before = white_count
            else:
                if cont_increase < break_parameter:
                    if white_count < before:
                        before = white_count
                        cont_increase += 1
                else:
                    if cont_decrease == break_parameter:
                        if count:
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


# Function to draw vertical lines on the image
def draw_lines_on_image(draw, points_of_interest, height):
    for point in points_of_interest:
        draw.line((point, 0, point, height), fill="black", width=1)


# Function to draw the middle vertical line
def draw_middle_line(draw, middle_point, height):
    draw.line((middle_point, 0, middle_point, height), fill="red", width=2)


# Main function to process the image and save the result
def draw_lines_and_save(img, img_path, output_path):
    numpy_array = np.array(img)
    non_black_pixels_mask = numpy_array > 0

    height, width = non_black_pixels_mask.shape
    middle_point = find_middle_point(non_black_pixels_mask)

    points_of_interest_left = find_interesting_points(non_black_pixels_mask, middle_point)
    points_of_interest_right = find_interesting_points(non_black_pixels_mask[:, ::-1], width - middle_point)

    img_with_lines = img.copy()
    draw = ImageDraw.Draw(img_with_lines)

    draw_lines_on_image(draw, points_of_interest_left, height)
    draw_middle_line(draw, middle_point, height)

    for point in points_of_interest_right:
        new_point = width - point
        draw.line((new_point, 0, new_point, height), fill="black", width=1)

    img_with_lines.save(output_path)


if __name__ == "__main__":
    coco_json_path = r'C:\Users\edayu\PycharmProjects\Dental\DentalProject\dental\dental\gazi_gum_project\formatted_file.json'

    coco_data = load_coco_json(coco_json_path)

    for image_data in coco_data['images']:
        target_image_id = image_data['id']
        target_image_data = next((img for img in coco_data['images'] if img['id'] == target_image_id), None)
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == target_image_id]
        if target_image_data:
            mask = create_mask(target_image_data, annotations)
            mask_save_path = save_mask(mask, target_image_data["file_name"])
            img = Image.open(mask_save_path)
            output_path = f'local/masked_image_with_lines_{target_image_id}.bmp'
            draw_lines_and_save(img, mask_save_path, output_path)
        else:
            print(f"Image with ID {target_image_id} not found.")
