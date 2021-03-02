"""Replay manual annotation output to generate bit masks
"""

import manual_annotator

wells = # ML_training_images
training_data = # training_data.csv

for well, training_row in zip(wells, training_data) # synchronise well images and rows of training data
    # parse the training data
    contour_thickness, white_noise, min_area_thresh, max_area_thresh, growth_area = training_row

    contours = manual_annotator.get_contours(well, contour_thickness, white_noise, min_area_thresh, max_area_thresh):

    # resolve whether user has removed contours
    if len(contours) != n_contours:
        discrepancy = growth_area - sum(contours) # what is the missing area
        for x in choose(n_contours - contours, contours):
            # choose sets of contours from the amount that are missing
            sum(x) == discrepancy?

    contours = the proper set of contours

    # generate the mask by drawing and filling the contours:
    mask = np.zeros(well.shape) # make empty bitmask
    for contour in contours:
        cv.fillPoly(mask, pts=[contour], color=(255,255,255)) # or make it monochrome if you want a numpy bit matrix

    # write out the bitmask as png maybe?
    cv.imwrite(blah blah, mask)

    # or just yield the image as a numpy matrix?
    print(mask)
