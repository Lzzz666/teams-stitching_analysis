import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_seperated_lines():
    pass


def find_contours_of_color_bar(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 畫在image上
    print(f"found {len(contours)} contours")
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite('assets/output/contours.png', image)
    return contours

def get_mask(hsv, lower, upper):
    return cv2.inRange(hsv, lower, upper)

def get_color_bar_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定義遮罩
    mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 60, 60]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([160, 60, 60]), np.array([179, 255, 255]))
            )
    mask_green = get_mask(hsv, np.array([40, 60, 60]), np.array([80, 255, 255]))
    mask_blue = get_mask(hsv, np.array([100, 60, 60]), np.array([130, 255, 255]))
    mask_gray = get_mask(hsv, np.array([0, 0, 60]), np.array([180, 50, 200]))  # 灰色低飽和

    # 做 Open 運算移除灰色 mask 的雜點
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask_gray = np.zeros_like(cleaned_gray)
        cv2.drawContours(clean_mask_gray, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    else:
        clean_mask_gray = np.zeros_like(mask_gray)

    masks = {
        "red": mask_red,
        "green": mask_green,
        "blue": mask_blue,
        "gray": clean_mask_gray
    }

    return masks

def get_seperated_color_bar_rgb(image, masks):
    # 針對紅色 綠色 藍色 灰色 分別用 mask 過濾 image
    for color, mask in masks.items():
        image_color = cv2.bitwise_and(image, image, mask=mask)
        _, thresh = cv2.threshold(image_color, 10, 255, cv2.THRESH_BINARY)
        center_line_points = []  # 中線 (取平均值)
        upper_line_points = []   # 上緣 (取最小 y)
        for x in range(thresh.shape[1]):
            y_coords = np.where(thresh[:, x] == 255)[0]
            
            if len(y_coords) > 0:
                # 1. 中線位置
                y_mid = int(np.mean(y_coords))
                center_line_points.append((x, y_mid))

                # 2. 上緣位置 (最小 y，即色條最上方)
                y_top = int(np.min(y_coords))
                upper_line_points.append((x, y_top))

        distances = [] # x 軸: 距離 (直接使用 x 座標)
        blue_channel = []
        green_channel = []
        red_channel = []

        if len(center_line_points) > 1:
            for i in range(len(center_line_points) - 1):
                # TODO: get rgb color
                # 並且分通道 plot 出來中線經過的每個點的 rgb 值 (x 座標是 distance, y 座標是 color)
                # 畫出中線
                distances.append(center_line_points[i][0])
                blue_channel.append(image_color[center_line_points[i][1], center_line_points[i][0], 0])
                green_channel.append(image_color[center_line_points[i][1], center_line_points[i][0], 1])
                red_channel.append(image_color[center_line_points[i][1], center_line_points[i][0], 2])

        # Draw upper-edge rough preview (optional)
        if len(upper_line_points) > 1:
            for i in range(len(upper_line_points) - 1):
                cv2.line(image, upper_line_points[i], upper_line_points[i+1], (0, 200, 0), 1)

        # --- Smooth approximation of the upper edge ---------------------------------
        if len(upper_line_points) >= 4:
            pts = np.array(upper_line_points)
            x_pts, y_pts = pts[:, 0], pts[:, 1]

            # Choose polynomial degree: quadratic for few points, quartic for more
            degree = 4 if len(pts) > 6 else 2
            coeffs = np.polyfit(x_pts, y_pts, degree)

            # Generate dense x-coordinates and evaluate polynomial for smooth y
            x_dense = np.linspace(x_pts.min(), x_pts.max(), max(100, len(x_pts) * 3))
            y_dense = np.polyval(coeffs, x_dense)

            smooth_curve = np.stack((x_dense, y_dense), axis=-1).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [smooth_curve], isClosed=False, color=(0, 255, 0), thickness=2)

        # plot 出 上緣逼近的值，也 plot 出原始的上緣值 比較兩條差異

        # 畫出上緣逼近的值，也 plot 出原始的上緣值 比較兩條差異
        plt.plot(x_pts, y_pts, '-', label='Original Points', color='red')
        plt.plot(x_dense, y_dense, 'b-', label='Smoothed Curve')
        plt.legend()
        plt.show()

        # 在同一個 Figure 中使用三個子圖（由上到下分別畫 R、G、B）
        fig, (ax_r, ax_g, ax_b) = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
        fig.suptitle('RGB value comparison of the left and right lines', fontsize=16, fontweight='bold')
        ax_r.plot(distances, red_channel, color='red')
        ax_r.set_ylabel('R')
        ax_g.plot(distances, green_channel, color='green')
        ax_g.set_ylabel('G')
        ax_b.plot(distances, blue_channel, color='blue')
        ax_b.set_ylabel('B')
        ax_b.set_xlabel('Distance')
        ax_r.set_ylim(0, 255)
        ax_g.set_ylim(0, 255)
        ax_b.set_ylim(0, 255)
        plt.tight_layout()
        plt.show()
        cv2.imshow('Result', image)
        cv2.imshow(color, image_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_seperated_edges(image, masks):
    # image: grayscale ROI
    # 建立一個彩色版本的灰階 ROI，方便後續以不同顏色畫出輪廓
    edges_visual = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    # 預先定義不同遮罩對應的顏色，用來在視覺化圖上標示
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "gray": (128, 128, 128)
    }

    for color, mask in masks.items():
        # 依據遮罩擷取該區域的灰階 ROI
        region = cv2.bitwise_and(image, image, mask=mask)

        # 先做模糊降低雜訊，再用 Otsu 自動閾值二值化
        blurred = cv2.GaussianBlur(region, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形態 Open 移除小雜訊
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        # 取最大輪廓（假設顏色條是最大的物體）
        largest = max(contours, key=cv2.contourArea)

        # 在視覺化圖上畫出輪廓
        draw_color = color_map.get(color, (255, 255, 255))
        cv2.drawContours(edges_visual, [largest], -1, draw_color, 2)

        # 亦可將各顏色的二值化結果另存，方便除錯
        cv2.imwrite(f"assets/output/{color}_edge.png", binary)

    # 將所有輪廓彙整後輸出/顯示
    cv2.imwrite("assets/output/roi_edges.png", edges_visual)
    cv2.imshow("Separated Edges", edges_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

