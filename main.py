from utils import load_image
from marker_processor import find_marker
from image_processor import find_contours_of_color_bar, get_color_bar_mask, get_seperated_color_bar_rgb, get_seperated_edges
import cv2

def main():
    image = load_image()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    center_of_marker = find_marker(gray_image)
    # TODO: 根據 4 個點，畫出一個矩形 然後成為新的 ROI
    sorted_center_of_marker = sorted(center_of_marker, key=lambda x: x[0])
    left_top = (sorted_center_of_marker[0][0], sorted_center_of_marker[0][1])
    right_bottom = (sorted_center_of_marker[3][0], sorted_center_of_marker[3][1])

    roi = image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    find_contours_of_color_bar(roi_gray)
    masks = get_color_bar_mask(roi)
    get_seperated_color_bar_rgb(roi, masks)
    get_seperated_edges(roi_gray,masks)


if __name__ == "__main__":
    main() 