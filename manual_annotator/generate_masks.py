"""Replay manual annotation output to generate bit masks

you can flip and rotate the masks for 8x the training data here too
"""

import manual_annotator

wells = # ML_training_images
training_data = # training_data.csv

for well, training_row in zip(wells, training_data) # synchronise well images and rows of training data
    # parse the training data
    contour_thickness, white_noise, min_area_thresh, max_area_thresh, growth_area, n_contours = training_row

    def sum(contours):
        # note that our definition of growth_area considers min/max area thresholds
        total = 0
        for contour in contours
            area = cv.contourArea(contour) # calculate area of contour
            if area > max thresh
                continue
            if area < min thresh
                continue
            total += area
        return total


    contours = manual_annotator.get_contours(well, contour_thickness, white_noise, min_area_thresh, max_area_thresh):

    # resolve whether user has removed contours
    if len(contours) != n_contours:
        discrepancy = growth_area - sum(contours) # what is the missing area
        for m_subset in choose(len(contours) - n_contours, contours):
            # choose sets of m contours from the amount that are missing
            sum(m_subset) == discrepancy?
                remove these m contours from contours

    # generate the mask by drawing and filling the contours:
    mask = np.zeros(well.shape) # make empty bitmask
    for contour in contours:
        cv.fillPoly(mask, pts=[contour], color=(255,255,255)) # or make it monochrome if you want a numpy bit matrix

    # write out the bitmask as png maybe?
    cv.imwrite(blah blah, mask)

    # or just yield the image as a numpy matrix?
    print(mask)
