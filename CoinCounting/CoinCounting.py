import cv2
import numpy as np



# PREPROCESS
def normalize_with_background(image, scale):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    bg = np.clip(bg, 1, 255)
    bg_color = cv2.merge([bg] * 3)

    norm = (image / bg_color) * scale
    return np.clip(norm, 0, 255).astype(np.uint8)



# YELLOW 
def yellow_processing(image):
    norm = normalize_with_background(image, 255)

    mask = cv2.inRange(
        norm, (20, 240, 240), (150, 255, 255)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    return mask, contours



#  BLUE 
def blue_processing(image):
    #ใช้ 255 แล้วมีบางเหรียญที่ติดกันแล้วมันนับเป็นเหรียญเดียวกัน
    norm = normalize_with_background(image, 190)

    #ถ้าไม่ทำแยกบางเหรียญที่ติดกันไม่ได้ ใช้เป็น opening เลยไม่ได้เพราะใช้ kernel ไม่เท่ากัน
    norm = cv2.erode(norm, np.ones((22, 22), np.uint8))
    norm = cv2.dilate(norm, np.ones((5, 1), np.uint8))

    mask = cv2.inRange(
        norm, (175, 145, 0), (255, 255, 155)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    return mask, contours



#   WATERSHED 
def split_with_watershed(binary_mask, image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opening = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, kernel, iterations=2
    )

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    _, fg = cv2.threshold(
        dist, 0.1 * dist.max(), 255, cv2.THRESH_BINARY
    )
    fg = np.uint8(fg)

    bg = cv2.dilate(opening, kernel, iterations=1)
    unknown = cv2.subtract(bg, fg)

    _, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[unknown == 255] = 0

    return cv2.watershed(image, markers)



#  COUNTING 
def count_coins(contours, mask, image, area_thresh, min_area, draw_color):
    count = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        single = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(single, [c], -1, 255, -1)

        if area > area_thresh:
            markers = split_with_watershed(single, image)
            count += len(np.unique(markers)) - 2
        else:
            count += 1

        cv2.drawContours(image, [c], -1, draw_color, 2)

    return count



#  MAIN
def coinCounting(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read image:", image_path)
        return [0, 0]

    img = cv2.resize(img, (500, 500))
    result = img.copy()

    y_mask, y_contours = yellow_processing(img)
    b_mask, b_contours = blue_processing(img)

    yellow_count = count_coins(
        y_contours, y_mask, result, 1500, 50, (0, 255, 255)
    )
    blue_count = count_coins(
        b_contours, b_mask, result, 1500, 45, (255, 0, 0)
    )

    cv2.putText(
        result,
        f"[{yellow_count} , {blue_count}]", 
        (50, 80), 
        cv2.FONT_HERSHEY_PLAIN, 
        2, 
        (255, 0, 255), 
        2 
    )

    cv2.imshow("Final Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Yellow Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Blue Mask", cv2.WINDOW_NORMAL)
    
    cv2.imshow("Final Result", result)
    cv2.imshow("Yellow Mask", y_mask) 
    cv2.imshow("Blue Mask", b_mask)
    
    cv2.moveWindow("Final Result", 50, 50) 
    cv2.moveWindow("Yellow Mask", 550, 50) 
    cv2.moveWindow("Blue Mask", 1050, 50)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [yellow_count, blue_count]


#   RUN DATASET
for i in range(1, 11):
    print(i, coinCounting(f"./coin{i}.jpg"))
