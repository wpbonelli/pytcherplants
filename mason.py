import argparse
import csv
from glob import glob
from os.path import join
from pathlib import Path

import cv2
import numpy as np
from tabulate import tabulate


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def analyze_colors(cluster):
    centroids = cluster.cluster_centers_
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    rect = np.zeros((200, 1200, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])[0:-1]
    percents = [percent for (percent, color) in colors]
    normalized_percents = [float(percent) / sum(percents) for percent in percents]
    normalized_colors = [(percent, color, RGB2HEX(color)) for (percent, color) in zip(normalized_percents, [c for (p, c) in colors])]
    start = 0

    for (percent, color, hex) in normalized_colors:
        text = f"{hex}: {round(percent * 100, 2)}%"
        print(text)
        end = start + (percent * 1200)
        x = int(start)
        y = 0
        cv2.rectangle(rect, (x, y), (int(end), 200), color.astype("uint8").tolist(), -1)
        cv2.putText(rect, text, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        start = end

    return rect, normalized_colors


def color_analysis_1(image, i, output_directory, base_name):
    Z = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((image.shape[:-1]))
    reduced = np.uint8(centers)[labels]

    print(f"Contour {i} color distribution:")

    for ii, cc in enumerate(centers):
        mask = cv2.inRange(labels, ii, ii)
        mask = np.dstack([mask] * 3)  # Make it 3 channel
        ex_reduced = cv2.bitwise_and(reduced, mask)
        cv2.imwrite(join(output_directory, base_name + '.contour' + str(i) + '.cluster' + str(ii) + '.png'), ex_reduced)


def color_analysis_2(image, i, output_directory, base_name):
    reshaped = image.reshape((image.shape[0] * image.shape[1], 3))
    from sklearn.cluster import KMeans
    clu = KMeans(n_clusters=6).fit(reshaped)
    print(f"Contour {i} color distribution:")
    visualize, colors = analyze_colors(clu)
    cv2.imwrite(join(output_directory, base_name + '.contour' + str(i) + '.colors.png'), visualize)
    colors_csv_path = join(output_directory, f"{base_name}.contour{str(i)}.colors.csv")
    with open(colors_csv_path, 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Proportion', 'Color (RGB)', 'Color (Hex)'])
        for c in colors: writer.writerow([c[0], c[1], c[2]])


def contour(orig, thresh, base_name, output_directory, i, c):
    img_height, img_width, img_channels = orig.shape
    orig_copy = orig.copy()

    # get bounding rect
    x, y, w, h = cv2.boundingRect(c)

    if w > img_width * 0.05 and h > img_height * 0.05:
        print(f"Contour {i} width and height: {w}, {h}")
        roi = orig[y:y + h, x:x + w]
        trait_img = cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 255, 0), 3)  # draw a green rectangle to visualize the bounding rect

        # get mask
        maskk = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(maskk, [c], 0, 255, -1)
        masked = cv2.bitwise_and(orig_copy, orig_copy, mask=maskk)

        # get convex hull
        hull = cv2.convexHull(c)
        trait_img = cv2.drawContours(orig, [hull], -1, (0, 0, 255), 3)

        # calculate area
        area = cv2.contourArea(c)
        print(f"Contour {i} area: {round(area, 2)}")

        # calculate convex hull area and solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        print(f"Contour {i} solidity: {round(solidity, 2)}")
        cv2.imwrite(join(output_directory, base_name + '.contour' + str(i) + '.mask.png'), masked)

        # color analysis
        # color_analysis_1(masked, i, output_directory, base_name)
        color_analysis_2(masked.copy(), i, output_directory, base_name)

        return i, area, solidity, w, h, x, y, x + w, y + h
    else:
        print(f"Contour {i} too small, skipping")
        return i, None, None, None, None, None, None, None, None


def contours(orig, thresh, base_name, save_path, keep=6):
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contour(s), keeping top {keep}")

    sorted_contours = list(reversed(sorted(contours, key=cv2.contourArea)))
    top = sorted_contours[0:keep]
    args = [(orig, thresh, base_name, save_path, i, c) for (i, c) in enumerate(top)]
    results = []
    for arg in args: results.append(contour(*arg))

    headers = ['Index', 'Area', 'Solidity', 'Width', 'Height', 'BBoxStartX', 'BBoxStartY', 'BBoxEndX', 'BBoxEndY']
    results = [result for result in results if result[1] is not None]  # remove null rows
    table = tabulate(results, headers=headers, tablefmt='orgtbl')
    print(table)

    contours_csv_path = join(save_path, f"{base_name}.contours.csv")
    with open(contours_csv_path, 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for r in results: writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]])

    trait_img = cv2.drawContours(orig, contours, -1, (255, 255, 0), 1)
    return trait_img, max([r[1] for r in results])


def remove_grays_1(image, base_name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    thresh1 = cv2.threshold(s, 90, 255, cv2.THRESH_TRIANGLE)[1]
    thresh2 = cv2.threshold(v, 2, 255, cv2.THRESH_TRIANGLE)[1]
    thresh2 = 255 - thresh2
    hsv_mask = cv2.add(thresh1, thresh2)
    # cv2.imwrite(join(output, base_name + '.prehsvmask.png'), hsv_mask)
    hsv_result = image.copy()
    hsv_result[hsv_mask == 0] = (0, 0, 0)
    # cv2.imwrite(join(output, base_name + '.prenograys.png'), hsv_result)
    # cv2.imwrite(join(output, base_name + '.premasked.png'), hsv_result)
    return hsv_result


def remove_grays_2(image, opened, base_name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 5, 50], np.uint8)
    upper_gray = np.array([179, 50, 255], np.uint8)
    _, mask_gray_inv = cv2.threshold(
        cv2.cvtColor(cv2.cvtColor(cv2.bitwise_and(image, image, mask=cv2.inRange(hsv, lower_gray, upper_gray)), cv2.COLOR_HSV2RGB),
                     cv2.COLOR_RGB2GRAY),
        1, 255, cv2.THRESH_BINARY)
    mask_gray = opened - mask_gray_inv
    # cv2.imwrite(join(output, base_name + '.mask.gray.inv.png'), mask_gray_inv)
    # cv2.imwrite(join(output, base_name + '.mask.gray.png'), mask_gray)
    return mask_gray


def remove_grays_3(image, base_name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    thresh1 = cv2.threshold(s, 90, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.threshold(v, 2, 255, cv2.THRESH_BINARY)[1]
    thresh2 = 255 - thresh2
    hsv_mask = cv2.add(thresh1, thresh2)
    cv2.imwrite(join(output, base_name + '.hsvmask2.png'), hsv_mask)
    hsv_result = image.copy()
    hsv_result[hsv_mask == 0] = (0, 0, 0)
    cv2.imwrite(join(output, base_name + '.nograys2.png'), hsv_result)
    return hsv_result


def mask_1(image, base_name):
    print("Creating mask (strategy 1)")

    # gaussian blur
    blur = cv2.GaussianBlur(image, (7, 7), cv2.BORDER_DEFAULT)

    # remove background grays (1st pass)
    hsv_result = remove_grays_1(blur, base_name)

    # binary threshold
    gray = cv2.cvtColor(hsv_result, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2., cv2.THRESH_BINARY, 199, 5)
    # cv2.imwrite(join(output, base_name + '.binary.png'), binary)

    # opening (1st pass)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # cv2.imwrite(join(output, base_name + '.opened.png'), opened)

    # gray/black background mask (2nd pass)
    # mask_gray = remove_grays_2(image, opened, base_name)

    # opening + dilation (2nd pass)
    # reopened = cv2.dilate(opened, np.ones((5, 5), np.uint8)) # cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # cv2.imwrite(join(output, base_name + '.reopened.png'), reopened)

    # intermediate mask
    masked = cv2.bitwise_and(image, image, mask=opened)
    cv2.imwrite(join(output, base_name + '.masked.png'), masked)
    return opened, masked


def mask_2(image, base_name):
    print("Creating mask (strategy 2)")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 70, 40])
    upper_blue = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dilated = cv2.dilate(opened, np.ones((5, 5)))

    masked = cv2.bitwise_and(image, image, mask=dilated)
    cv2.imwrite(join(output, base_name + '.masked.png'), masked)
    return opened, masked


def process(input, output, base_name, count):
    image = cv2.imread(input)

    # mask1, masked1 = mask_1(image, base_name)
    mask, masked = mask_2(image, base_name)

    # find contours
    print("Finding contours")
    trait_img, max_area = contours(masked, mask, base_name, output, count)
    cv2.imwrite(join(output, base_name + '.contours.png'), trait_img)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input path")
    ap.add_argument("-o", "--output", required=True, help="Output path")
    ap.add_argument("-ft", "--filetypes", required=False, default='png,jpg', help="Image filetypes")
    ap.add_argument("-c", "--count", required=False, default=6, help="Number of individuals")

    args = vars(ap.parse_args())
    input = args['input']
    output = args['output']
    extensions = args['filetypes'].split(',') if 'filetypes' in args else []
    extensions = [e for es in [[extension.lower(), extension.upper()] for extension in extensions] for e in es]
    patterns = [join(input, f"*.{p}") for p in extensions]
    files = sorted([f for fs in [glob(pattern) for pattern in patterns] for f in fs])
    count = int(args['count'])

    if Path(input).is_dir():
        print(f"Found {len(files)} images")
        for file in files:
            print(f"Processing {file}")
            process(file, output, Path(file).stem, count)
    else:
        print(f"Processing {input}")
        process(input, output, Path(input).stem, count)
