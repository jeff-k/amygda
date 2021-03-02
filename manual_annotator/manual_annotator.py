import numpy as np
import cv2 as cv
import pandas as pd
import argparse
from pathlib import Path

GAME_TITLE = 'Bug Lasso'

hotkeys_text = """
Hotkeys:
[ENTER] or [N] = Confirm and next ("I found the correct answer")
[F] = Fail and next ("I could not find the correct answer")
[P] = Go back (previous)
[B] = Flag bubbles
[C] = Flag condensation
[D] = Flag a dry well
[LEFT MOUSE CLICK] = Remove a contour
[RIGHT MOUSE CLICK] = Clear all removed contours
[H] = Show hotkey menu
[ESC] = Save and exit
"""

colors=[(255,0,0), (0,255,0), (0,0,255),
            (255,255,0), (255,0,255),
            (0,255,255)]

# TODO: control multithreaded access?
good_contours=[]
forbidden_contours=[]


def get_color(color_index):
    if color_index == len(colors):
        color_index = 0
    color = colors[color_index]
    color_index+=1
    return color_index, color


def save_rows(calls_csv, out_file):
    cv.displayOverlay(GAME_TITLE, f"Saving calls to {out_file}...", 1000)
    calls_csv.to_csv(out_file)
    cv.displayOverlay(GAME_TITLE, f"Calls successfully saved to {out_file}!", 1000)


def null_fn(x):
    pass

def mouse_callback(event, x, y, flags, params):
    global good_contours, forbidden_contours
    if event==1:
        for cnt in good_contours:
            if cv.pointPolygonTest(cnt, (x,y), False) >= 0:
                if not is_a_forbidden_contours(cnt):
                    forbidden_contours.append(cnt)
    elif event==2:
        forbidden_contours = []


def is_a_forbidden_contours(cnt):
    global forbidden_contours
    return any(np.array_equal(cnt, x) for x in forbidden_contours)


def get_args():
    parser = argparse.ArgumentParser(description='Manual annotator for a list of well images.')
    parser.add_argument('--wells_csv', type=str, help='CSV file with a list of wells. This file is the output of create_participants_dataset.py script', required=True)
    parser.add_argument('--output_csv', type=str, help='The output CSV file', required=True)
    args = parser.parse_args()
    return args

def window_is_closed():
    return cv.getWindowProperty(GAME_TITLE, cv.WND_PROP_VISIBLE) == 0.0

def maximize_window():
    cv.setWindowProperty(GAME_TITLE, cv.WND_PROP_FULLSCREEN, 1.0)

def show_images(human_vision, static_image):
    image_to_show = np.hstack((human_vision, static_image))
    cv.imshow(GAME_TITLE, image_to_show)


def mask_circles(well_gray_with_border, well_shadow):
    circs = cv.HoughCircles(well_gray_with_border, cv.HOUGH_GRADIENT, 1, 1, param1=20, param2=well_shadow, minRadius=37,
                            maxRadius=42)
    if circs is not None:
        for cs in circs:
            for c in cs:
                x, y, r = c
                r = int(r)
                x = int(x)
                y = int(y)
                mask = np.zeros(well_gray_with_border.shape, np.uint8)
                cv.circle(mask, (x, y), r, 255, thickness=-1)
                well_gray_with_border = cv.bitwise_and(well_gray_with_border, well_gray_with_border, mask=mask)
    return well_gray_with_border


def load_dataframe(output_csv_filepath):
    output_csv_path = Path(output_csv_filepath)
    if output_csv_path.exists():
        return pd.read_csv(output_csv_path, index_col="well_path")
    else:
        calls_csv = pd.DataFrame(
            columns=["contour_thickness","white_noise","min_area_threshold","max_area_threshold",
                     "well_shadow","growth","nb_of_contours","pass_or_fail", "flags"])
        calls_csv.index.name = "well_path"
        return calls_csv

def get_contours(well, contour_thickness, white_noise, min_area_thresh, max_area_thresh):

    well_gray_with_border = cv.cvtColor(well, cv.COLOR_BGR2GRAY)
    well_gray_with_border = cv.copyMakeBorder(well_gray_with_border, b, b, b, b, cv.BORDER_CONSTANT, 0)
    well_gray_with_border = mask_circles(well_gray_with_border, well_shadow)
    binarized_image = well_gray_with_border.copy()
    binarized_image = cv.adaptiveThreshold(binarized_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, contour_thickness, white_noise)
    static_well_image = well.copy()

    well_with_border = cv.copyMakeBorder(well, b, b, b, b, cv.BORDER_CONSTANT, 0)
    static_well_with_border = cv.copyMakeBorder(well, b, b, b, b, cv.BORDER_CONSTANT, 0)
    binarized_image_with_color = cv.cvtColor(binarized_image, cv.COLOR_GRAY2BGR)

    # find and draw contours
    contours = cv.findContours(binarized_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
    total_area = 0
    n_contours = 0
    color_index = 0

    good_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        contour_has_good_area = area >= min_area_thresh and area <= max_area_thresh
        if contour_has_good_area and not is_a_forbidden_contours(cnt):
            color_index, color = get_color(color_index)
            cv.drawContours(well_with_border, [cnt], -1, color, 1)
            total_area += area
            n_contours += 1
            good_contours.append(cnt)
    return good_contours


if __name__ == "__main__":
    args = get_args()
    wells_paths = pd.read_csv(args.wells_csv)["wells"]

    cv.namedWindow(GAME_TITLE, cv.WINDOW_GUI_NORMAL)
    # maximize_window()
    cv.setMouseCallback(GAME_TITLE, mouse_callback)

    cv.createTrackbar('Contour thickness', GAME_TITLE, 3, 100, null_fn)
    cv.createTrackbar('White noise', GAME_TITLE, 3, 50, null_fn)
    cv.createTrackbar('Well shadow', GAME_TITLE, 10, 70, null_fn)
    cv.createTrackbar('Min growth', GAME_TITLE, 0, 500, null_fn)
    cv.createTrackbar('Max growth', GAME_TITLE, 0, 5000, null_fn)


    # default positions
    cv.setTrackbarPos('Contour thickness', GAME_TITLE, 17)
    cv.setTrackbarPos('White noise', GAME_TITLE, 6)
    cv.setTrackbarPos('Min growth', GAME_TITLE, 28)
    cv.setTrackbarPos('Max growth', GAME_TITLE, 1000)
    cv.setTrackbarPos('Well shadow', GAME_TITLE, 14)

    calls_csv = load_dataframe(args.output_csv)
    well_no = len(calls_csv)
    iterations = 0
    while well_no < len(wells_paths) and not window_is_closed():
        iterations+=1

        save_automatically = iterations % 5 == 0
        if save_automatically:
            save_rows(calls_csv, args.output_csv)
        else:
            cv.displayOverlay(GAME_TITLE, f"Well {well_no+1}", 1000)
        well_path = wells_paths[well_no]
        well = cv.imread(well_path)
        flags = set([])
        forbidden_contours = []

        while True and not window_is_closed():
            # get input
            well_shadow = cv.getTrackbarPos('Well shadow', GAME_TITLE)
            if well_shadow < 1:
                well_shadow=1
            b = 40
            contour_thickness = cv.getTrackbarPos('Contour thickness', GAME_TITLE)
            if contour_thickness % 2 != 1:
                contour_thickness += 1
            if contour_thickness < 3:
                contour_thickness = 3
            white_noise = cv.getTrackbarPos('White noise', GAME_TITLE)
            min_area_thresh = cv.getTrackbarPos('Min growth', GAME_TITLE)
            max_area_thresh = cv.getTrackbarPos('Max growth', GAME_TITLE)


            good_contours = get_contours(well, contour_thickness, white_noise, min_area_thresh, max_area_thresh)


            # write info to images
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(well_with_border, f"Growth: {int(total_area)}", (0, well_with_border.shape[1] - 5), font, 0.7, (0, 255, 0), 1)  # , cv.LINE_AA)
            cv.putText(well_with_border, ','.join(flags), (0, 12), font, 0.5, (0, 0, 255), 1)  # , cv.LINE_AA)
            if len(forbidden_contours):
                cv.putText(well_with_border, f"{len(forbidden_contours)} REMOVED", (0, 28), font, 0.5, (0, 0, 255), 1)  # , cv.LINE_AA)

            show_images(well_with_border, static_well_with_border)


            key = cv.waitKey(25)
            if key == 27: # esc
                save_rows(calls_csv, args.output_csv)
                exit()
            elif key == ord('b') or key == ord('B'):
                if 'BUBBLE' in flags:
                    flags.remove('BUBBLE')
                else:
                    flags.add('BUBBLE')
            elif key == ord('c') or key == ord('C'):
                if 'COND' in flags:
                    flags.remove('COND')
                else:
                    flags.add('COND')
            elif key == ord('d') or key == ord('D'):
                if 'DRY' in flags:
                    flags.remove('DRY')
                else:
                    flags.add('DRY')
            elif key == 13 or key == ord('n') or key == ord('N'): # enter
                flags = ':'.join(flags)
                calls_csv.loc[well_path] = [contour_thickness, white_noise, min_area_thresh, max_area_thresh, well_shadow, total_area, n_contours, "PASS", flags]
                well_no += 1
                break
            elif key == ord('f') or key == ord('F'):
                flags = ':'.join(flags)
                calls_csv.loc[well_path] = [contour_thickness, white_noise, min_area_thresh, max_area_thresh, well_shadow, total_area, n_contours, "FAIL", flags]
                well_no += 1
                break
            elif key == ord('p') or key == ord('P'):
                if well_no > 0: well_no -= 1
                break
            elif key == ord('h') or key == ord('H'):
                cv.displayOverlay(GAME_TITLE, hotkeys_text.upper(), 10000)
            elif key == ord('s') or key == ord('S'):
                save_rows(calls_csv, args.output_csv)

    save_rows(calls_csv, args.output_csv)
    cv.destroyAllWindows()

    if len(wells_paths) == len(calls_csv):
        print("All done, thanks!")

