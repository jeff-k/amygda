#! /usr/bin/env python

from amygda.core import PlateMeasurement
import argparse, numpy
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--image",help="the path to the image")
parser.add_argument("--growth_pixel_threshold",default=130,type=int,help="the pixel threshold, below which a pixel is considered to be growth (0-255, default=130)")
parser.add_argument("--growth_percentage",default=2,type=float,help="if the central measured region in a well has more than this percentage of pixels labelled as growing, then the well is classified as growth (default=2).")
parser.add_argument("--measured_region",default=0.5,type=float,help="the radius of the central measured circle, as a decimal proportion of the whole well (default=0.5).")
parser.add_argument("--sensitivity",default=4,type=float,help="if the average growth in the control wells is more than (sensitivity x growth_percentage), then consider growth down to this sensitivity (default=4)")
parser.add_argument("--file_ending",default="-raw",type=str,help="the ending of the input file that is stripped. Default is '-raw' ")
parser.add_argument("--plate_design",default="UKMYC5",type=str,help="the name of the plate design. Must have a series of matching files in config/")
# parser.add_argument("--pixel_intensities",action="store_true",help="calculate and store the measured pixel intensities in the centre of each well? Default is False")
options = parser.parse_args()

# parse the path to the input image and work out its stem of
if '/' in options.image:
    cols=options.image.split('/')
    image_name=cols[-1].split(options.file_ending)[0]
    image_path=cols[0]+"/"
    for i in cols[1:-1]:
        image_path+=i
        image_path+="/"
else:
    image_path="."
    image_name=options.image.split(options.file_ending)[0]

assert options.plate_design in ['UKMYC5','UKMYC6','GPALL1F','CHNMCMM2'], "this plate design is not recognised"

# create a new measurement
plate=PlateMeasurement(image_path,categories={'ImageFileName':image_name},configuration_path="config/",pixel_intensities=False,plate_design=options.plate_design)

# create the path for the output images
plate_stem=plate.abspath+plate.image_name

# define some colours for the output
# the below are a contrasting set of colours taken from ColorBrewer 2
# http://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=7
# you are, of course, free to make up your own!
# Note these are (B,G,R) not (R,G,B)
pink=(138,41,231)
yellow=(51,255,255)
teal=(119,158,27)
green=(36,166,102)
black=(0,0,0)
white=(255,255,255)

# load the raw image
plate.load_image("-raw.png")

# record that this image exists
plate.categories['IM_IMAGE_DOWNLOADED']=True

# apply a mean shift filter to smooth the background colours
plate.mean_shift_filter()

plate.save_image("-msf.png")

# apply the local histogram equalisation method to improve contrast
plate.equalise_histograms_locally()

plate.save_image("-clahe.png")

plate.stretch_histogram(debug=False)

# save the filtered image
plate.save_image("-filtered.png")

# record that this image has been filtered
plate.categories['IM_IMAGE_FILTERED']=True

# load in the photo of the plate
plate.load_image("-filtered.png")

# attempt to segment the wells
if plate.identify_wells(hough_param1=20,hough_param2=25,radius_tolerance=0.005,verbose=False):

    plate.categories['IM_WELLS_IDENTIFIED']=True

     # measure growth
    plate.measure_growth(region=options.measured_region,threshold_pixel=options.growth_pixel_threshold,threshold_percentage=options.growth_percentage,sensitivity=options.sensitivity)

    # save the numpy arrays containing the growth, positions of wells etc to disc
    plate.save_arrays("-arrays.npz")

    # draw circles around the wells
    plate.annotate_well_circumference(color=pink,linewidth=2)

    # write the drug and concentration
    plate.annotate_well_drugs_concs(color=black,fontsize=0.5)

    # add squares where the algorithm has detected growth
    # plate.annotate_well_analysed_region(growth_color=yellow,region=options.measured_region,thickness=3)

    # save the final image with wells with identified growth marked by red squares
    plate.save_image("-growth.png")

    # write the MICs to a simple plaintext file (they are stored in the JSON file but this is harder to read)
    plate.write_mics("-mics.txt")
