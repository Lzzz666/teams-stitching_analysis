import cv2
import numpy as np

def visualize_center(target, center):
    """mark the center point on the image"""
    result = target.copy()
    cv2.circle(result, center, 5, (0, 0, 255), -1)
    cv2.circle(result, center, 10, (0, 255, 0), 2)
    return result


def find_center_by_hough_lines(target):
    """find the center by hough lines (the intersection of the horizontal and vertical lines)"""
    edges = cv2.Canny(target, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    
    if lines is None:
        print("no lines found")
        return None, target
    
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        if abs(angle) < 10 or abs(angle) > 170:
            horizontal_lines.append(line[0])
        elif abs(abs(angle) - 90) < 10:
            vertical_lines.append(line[0])
    
    if not horizontal_lines or not vertical_lines:
        print("no enough horizontal or vertical lines")
        return None, target
    
    img_center_x, img_center_y = target.shape[1] // 2, target.shape[0] // 2
    
    best_h_line = min(horizontal_lines, key=lambda line: abs((line[1] + line[3]) / 2 - img_center_y))
    best_v_line = min(vertical_lines, key=lambda line: abs((line[0] + line[2]) / 2 - img_center_x))
    
    h_y = (best_h_line[1] + best_h_line[3]) / 2
    v_x = (best_v_line[0] + best_v_line[2]) / 2
    
    center = (int(v_x), int(h_y))
    result = visualize_center(target, center)
    
    return center, result

def extract_target_manually(image):
    print("=== select target region ===")
    print("Please drag to select a target region")
    

    display_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    try:
        print("Info: drag to select target region → Space/Enter confirm → ESC cancel")
        roi = cv2.selectROI("select target region", display_image, False, False)
        cv2.destroyAllWindows()
        
        x, y, w, h = roi
        
        if w > 0 and h > 0:
            target_region = image[y:y+h, x:x+w]
            cv2.imwrite('assets/output/extracted_target.png', target_region)
            print(f"✓ extracted target: position({x},{y}), size({w}x{h})")
            print("✓ saved to assets/output/extracted_target.png")
            return target_region
        else:
            print("⚠ invalid region")
            return None
    except Exception as e:
        print(f"⚠ failed to select target region: {e}")
        return None


def detect_correction_points(gray_image):
    print("\n" + "="*60)
    print("start to detect correction points...")
    
    try:
        print("\n--- pattern matching ---")
        corners = find_octagon_pattern_matching(gray_image)

        if corners and len(corners) >= 4:
            return corners
        else:
            print("⚠ pattern matching failed, try to manual marking...")
            
    except Exception as e:
        print(f"⚠ pattern matching failed: {e}")
        print("try to manual marking...")


def find_octagon_pattern_matching(image):
    print("=== pattern matching ===")
    
    target = extract_target_manually(image)
    if target is None:
        print("⚠ failed to get target template, cannot perform matching")
        return []

    h, w = target.shape
    
    if h > image.shape[0] or w > image.shape[1]:
        print("⚠ template size is too large, cannot perform matching")
        return []

    result = cv2.matchTemplate(image, target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    
    if max_val < 0.3:
        threshold = max_val * 0.8
        print(f"best matching score is too low, adjust threshold to: {threshold:.3f}")
    elif max_val < 0.6:
        threshold = 0.4
        print(f"use medium threshold: {threshold:.3f}")
    else:
        threshold = 0.6
        print(f"use standard threshold: {threshold:.3f}")
    
    locations = np.where(result >= threshold)

    print(f"found {len(locations[0])} matching positions")

    all_matches = []
    min_distance = min(w, h) // 2
    
    for y_pos, x_pos in zip(locations[0], locations[1]):
        match_score = result[y_pos, x_pos]
        all_matches.append((x_pos, y_pos, match_score))
    
    all_matches.sort(key=lambda x: x[2], reverse=True)
    
    filtered_matches = []
    
    for x_pos, y_pos, match_score in all_matches:
        is_duplicate = False
        for existing_x, existing_y, _ in filtered_matches:
            distance = np.sqrt((x_pos - existing_x)**2 + (y_pos - existing_y)**2)
            if distance < min_distance:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_matches.append((x_pos, y_pos, match_score))
    
    print(f"found {len(filtered_matches)} valid matches")
    
    result_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    all_corners = []
    center_of_octagon = []
    for i, (x, y, score) in enumerate(filtered_matches):
        print(f"match {i+1}: position({x},{y}), score: {score:.3f}")
        
        corners = [
            (x, y),           # top left
            (x + w, y),       # top right
            (x + w, y + h),   # bottom right
            (x, y + h)        # bottom left
        ]
        all_corners.extend(corners)
        
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_image, f'Match{i+1}: {score:.2f}', 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for j, (cx, cy) in enumerate(corners):
            cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result_image, f'P{j+1}', 
                       (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        roi = image[y:y+h, x:x+w]
        center, result = find_center_by_hough_lines(roi)
        print(f"center: {center}")

        if center:
            center_of_octagon.append((x+center[0], y+center[1]))
            cv2.circle(result_image, (x+center[0], y+center[1]), 5, (0, 0, 255), -1)
            cv2.putText(result_image, f'O{i+1}', 
                        (x+center[0]+5, y+center[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        else:
            print("⚠ failed to find center")
            #find  matching pattern 的中心
            center = (w // 2, h // 2)
            center_of_octagon.append((x+center[0], y+center[1]))
            cv2.circle(result_image, (x+center[0], y+center[1]), 5, (0, 0, 255), -1)
            cv2.putText(result_image, f'O{i+1}', 
                        (x+center[0]+5, y+center[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    if len(filtered_matches) < 4:
        print("⚠ found less than 4 corners, cannot perform calibration")
        return []

    cv2.imwrite('assets/output/pattern_matching_result.png', result_image)
    print(f"✓ pattern matching completed, found {len(all_corners)} corners")
    print("✓ result saved to assets/output/pattern_matching_result.png")
    
    return center_of_octagon

def find_marker(image):
    return find_octagon_pattern_matching(image)