from PIL import Image, ImageDraw
import numpy as np

def find_middle_point(non_black_pixels_mask, start_percentage=0.4, end_percentage=0.6):
    width = non_black_pixels_mask.shape[1]
    start = int(width * start_percentage)
    end = int(width * end_percentage)

    points_of_middle = []
    for item in range(start, end + 1):
        res_tmp = np.unique(non_black_pixels_mask[:, item], return_counts=True)
        if res_tmp[0][0] == True:
            points_of_middle.append(res_tmp[1][0])
        else:
            points_of_middle.append(res_tmp[1][1])

    middle_point = points_of_middle.index(max(points_of_middle)) + start
    return middle_point

def find_interesting_points(non_black_pixels_mask, middle_point, direction='left', break_parameter=6):
    points_of_interest = []
    for _ in range(3):
        cont_increase = 0
        cont_decrease = 0
        count = []
        index_number = 0
        for item in range(middle_point, -1, -1) if direction == 'left' else range(middle_point, non_black_pixels_mask.shape[1]):
            white_count = np.sum(non_black_pixels_mask[:, item])
            if cont_increase < break_parameter:
                if white_count < count[-1] if count else white_count:
                    count.append(white_count)
                    index_number = item
                    cont_increase += 1
                else:
                    count.clear()
                    cont_increase = 0
            else:
                if cont_decrease < break_parameter:
                    if white_count > count[-1] if count else white_count:
                        count.append(white_count)
                        cont_decrease += 1
                    else:
                        points_of_interest.append(index_number)
                        break
                else:
                    points_of_interest.append(index_number)
                    break
        if len(points_of_interest) != _ + 1:
            points_of_interest.append(index_number)
    return points_of_interest

def draw_lines_and_save(img_path, output_path):
    img = Image.open(img_path)
    numpy_array = np.array(img)
    non_black_pixels_mask = numpy_array > 0

    height, width = non_black_pixels_mask.shape
    middle_point = find_middle_point(non_black_pixels_mask)

    points_of_interest_left = find_interesting_points(non_black_pixels_mask, middle_point, direction='left')
    points_of_interest_right = find_interesting_points(non_black_pixels_mask, middle_point, direction='right')

    img_with_lines = img.copy()
    draw = ImageDraw.Draw(img_with_lines)

    # Orta noktayı çiz
    draw.line((middle_point, 0, middle_point, height), fill="red", width=1)

    # Sağ ve sol noktaları çiz
    for point in points_of_interest_left:
        draw.line((point, 0, point, height), fill="blue", width=1)
    for point in points_of_interest_right:
        draw.line((point, 0, point, height), fill="blue", width=1)

    img_with_lines.save(output_path)

if __name__ == "__main__":
    img_path = "/local/masks/2024-02-20 230833.png_mask.bmp"
    output_path = "local/masked_image_with_lines.bmp"
    draw_lines_and_save(img_path, output_path)
