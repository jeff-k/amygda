import glob
import numpy as np
import cv2 as cv
from pathlib import Path
import argparse
import pandas as pd
from scipy import stats


def score(radii):
    return np.var(radii)

def get_binned_index(binned_values, value, difference_threshold):
    for index, binned_value in enumerate(binned_values):
        if binned_value - difference_threshold <= value <= binned_value + difference_threshold:
            return index
    assert False, "Should not get here"

def get_circle_index(binned_x, binned_y, circle):
    x,y,r = circle[0], circle[1], circle[2]
    return get_binned_index(binned_x, x, r/2), get_binned_index(binned_y, y, r/2)


def append_if_there_is_no_value_close_enough(binned_values, value, difference_threshold):
    for binned_value in binned_values:
        if binned_value - difference_threshold <= value <= binned_value + difference_threshold:
            return
    binned_values.append(value)


def sort_circles(b_circs):
    binned_x = []
    binned_y = []
    for circle in b_circs:
        append_if_there_is_no_value_close_enough(binned_x, circle[0], circle[2]/2)
        append_if_there_is_no_value_close_enough(binned_y, circle[1], circle[2]/2)
    binned_x = sorted(binned_x)
    binned_y = sorted(binned_y)
    assert len(binned_x) == 12, "We should have 12 x values"
    assert len(binned_y) == 8, "We should have 8 x values"
    sorted_circles = [None] * 96

    for circle in b_circs:
        x,y = get_circle_index(binned_x, binned_y, circle)
        sorted_circles[y*12+x] = circle

    return sorted_circles

def get_circles(img):
    b_r = 2
    b_params = None
    b_circs = []
    for bs in range(3, 7, 2):
        for m in range(5, 10, 5):
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            grey = cv.GaussianBlur(grey, (bs, bs), m)

            for p1 in range(59, 70, 2):
                for p2 in range(10, 25, 1):
                    circs = cv.HoughCircles(grey, cv.HOUGH_GRADIENT, 1, 79, param1=p1, param2=p2, minRadius=38,
                                            maxRadius=43)

                    if circs is not None:
                        if len(circs[0]) == 96:
                            radii = np.array(list(map(lambda x: x[2], circs[0])))
                            s = score(radii)
                            if s < b_r:
                                b_r = s
                                b_params = (bs, m, p1, p2, np.average(radii))
                                b_circs = circs[0]

    b_circs = sort_circles(b_circs)
    return b_circs


def segment_image(cs, img):
    b = 40
    img = cv.copyMakeBorder(img, b, b, b, b, cv.BORDER_CONSTANT, 0)
    wells = []

    # wells are named in order: A1, A2, ..., B1, B2, ..., H12
    names = [f"{row}{column}" for row in "ABCDEFGH" for column in range(1, 13)]

    # The set of circles needs to be sorted in the same order!
    for name, c in zip(names, cs):
        x, y, r = c
        r = int(r)
        x = int(x + b)
        y = int(y + b)
        well = img[y - r:y + r, x - r:x + r]
        circle = np.zeros((r * 2, r * 2), np.uint8)
        cv.circle(circle, (r, r), r - 3, 255, thickness=-1)
        wells.append((name, cv.bitwise_and(well, well, mask=circle)))
    return wells


# image filtering based on original amygda
def stretch_histogram(image):
    mode=stats.mode(image,axis=None)[0]
    image=(np.array(image,dtype=np.int16))-mode

    lower=np.percentile(image,5)
    upper=np.percentile(image,95)

    pos_factor=40./upper
    neg_factor=-110./lower

    image=np.multiply(image,np.where(image>0,pos_factor,neg_factor))
    image=image+180.
    image=np.rint(image)
    image=np.uint8(image)

    return image


def filter_image(plate_image):
    # apply the mean shift filter
    plate_image = cv.pyrMeanShiftFiltering(plate_image, 10, 10)

    # equalise the image histogram locally (will take account of the uneven lighting)
    # note that the tile grid matches the wells
    plate_image = cv.cvtColor(plate_image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 12))
    plate_image = clahe.apply(plate_image)

    plate_image = stretch_histogram(plate_image)
    plate_image = cv.cvtColor(plate_image, cv.COLOR_GRAY2BGR)

    return plate_image


def split_plate_image_into_well_images_core(well_dir_for_this_plate, plate_image_filename, filter):
    plate_image = cv.imread(str(plate_image_filename))

    circles = get_circles(plate_image)

    if filter:
        plate_image = filter_image(plate_image)

    segments = segment_image(circles, plate_image)
    for well_label, segment in segments:
        cv.imwrite(str(well_dir_for_this_plate / f"{well_label}.png"), segment)


def remove_if_exists(dir_path):
    if dir_path.exists():
        dir_path.rmdir()

def split_plate_image_into_well_images(plate_image_filename, well_dir: Path, translation_csv: Path):
    dict_csv = {
        "original_plate_path_file": [],
        "anonymous_plate_path_dir_well_split": []
    }

    try:
        well_dir_filtered = Path(str(well_dir) + "_filtered")
        well_dir.mkdir(parents=True)
        well_dir_filtered.mkdir(parents=True)
        split_plate_image_into_well_images_core(well_dir, plate_image_filename, filter=False)
        split_plate_image_into_well_images_core(well_dir_filtered, plate_image_filename, filter=True)
        dict_csv["original_plate_path_file"].append(plate_image_filename.resolve())
        dict_csv["anonymous_plate_path_dir_well_split"].append(well_dir.resolve())
    except:
        pass # even if it fails, we still want to say it is ok, as nothing will be written to the csv

    df = pd.DataFrame.from_dict(dict_csv)
    df.to_csv(translation_csv, index=False)

def get_args():
    parser = argparse.ArgumentParser(description='Split plate images into well images.')
    parser.add_argument('--plate_image', type=str, help='Path to plate image', required=True)
    parser.add_argument('--well_dir', type=str, help='Directory with the individual well images', required=True)
    parser.add_argument('--translation_csv', type=str, help='Where to save the CSV with the translation of plate dir to well dir', required=True)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    plate_image_filename = Path(args.plate_image)
    well_dir = Path(args.well_dir)
    translation_csv = Path(args.translation_csv)
    split_plate_image_into_well_images(plate_image_filename, well_dir, translation_csv)

