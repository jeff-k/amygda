#! /usr/bin/env python

import pkg_resources, pickle

import cv2, numpy
from scipy import stats
import matplotlib.pyplot as plt

from datreant import Treant
import math

import os
from yattag import Doc
import shutil

from skimage.morphology import closing
import matplotlib.ticker as plticker


class PlateMeasurement(Treant):

    """ The PlateMeasurement class is a Treant for storing and analysing a
    photograph of a 96 well plate.

    Args:
        new (True/False): whether this is a new PlateMeasurement (default is False)
        plate_image (str): path to the image file to analyse
        categories (dict): a dictionary containing meta-data for the image. Must contain the name of the image (imagename) as {'ImageFileName':imagename}
        tags (str or list): strings containing tags to differentiate the Plates. Not required.
        well_dimensions (tuple): tuple of 2 integers defining the (rows, cols) of the plate. Default=(8,12)
    """

    _treanttype='PlateMeasurement'

    def __init__(self, plate_image, new=False, categories=None, tags='PlateMeasurement', well_dimensions=(8,12), configuration_path='config', plate_design=None,pixel_intensities=False):

        Treant.__init__(self, plate_image, categories=categories, tags=tags)

        # store the image name provided in the categories, if there is one
        if 'ImageFileName' in self.categories.keys():
            self.image_name=self.categories['ImageFileName']
        elif 'IMAGEFILENAME' in self.categories.keys():
            self.image_name=self.categories['IMAGEFILENAME']

        # store the (rows,cols) of the plate
        self.well_dimensions=well_dimensions

        # calculate and store the total number of wells
        self.number_of_wells = self.well_dimensions[0]*self.well_dimensions[1]

        # store the path and name of the plate
        self.configuration_path = '/'.join(('..',configuration_path))
        self.plate_design = plate_design

        # remember whether we will be analysing and keeping the pixel intensities etc
        self.pixel_intensities=pixel_intensities

        self.debug_images = f"{self.abspath}/debug_images/"
        os.makedirs(self.debug_images, exist_ok=True)

        self.histogram_files = []

        self.there_is_growth_dispersion_in_control_well = False
        self.apply_bubble_filter = False

    def load_image(self,file_ending):
        """ Load and store the image and its dimensions, then initialise a series of arrays to record the results
        """

        self.image_path=self.abspath+self.image_name+file_ending

        # load the image as a 3-channel array
        # it needs to be colour for the mean shift filter to work
        self.image = cv2.imread(self.image_path)

        # this is a colour image
        self.image_colour=True

        # determine the dimensions of the image
        self.image_dimensions=self.image.shape

        # calculate a float scaling factor assuming a standard image has 1000 pixels in the x dimension
        self.scaling_factor=self.image_dimensions[1]/1000.

        # create numpy arrays to store details about the wells
        self.well_index = numpy.zeros(self.well_dimensions)
        self.well_radii = numpy.zeros(self.well_dimensions)
        self.well_centre = numpy.zeros(self.well_dimensions,dtype=(int,2))
        self.well_top_left = numpy.zeros(self.well_dimensions,dtype=(int,2))
        self.well_bottom_right = numpy.zeros(self.well_dimensions,dtype=(int,2))
        self.well_growth = numpy.zeros(self.well_dimensions,dtype=numpy.float64)

        filename = pkg_resources.resource_filename("amygda", self.configuration_path+"/"+self.plate_design+"-drug-matrix.txt")
        self.well_drug_name = numpy.loadtxt(filename,delimiter=',',dtype=str)

        filename = pkg_resources.resource_filename("amygda", self.configuration_path+"/"+self.plate_design+"-conc-matrix.txt")
        self.well_drug_conc = numpy.loadtxt(filename,delimiter=',',dtype=float)

        filename = pkg_resources.resource_filename("amygda", self.configuration_path+"/"+self.plate_design+"-dilution-matrix.txt")
        self.well_drug_dilution = numpy.loadtxt(filename,delimiter=',',dtype=int)

        # identify the control wells
        self.well_positive_controls=[]
        for iy in range(0,self.well_dimensions[0]):
            for ix in range (0,self.well_dimensions[1]):
                if self.well_drug_conc[(iy,ix)]==0.0 and self.well_drug_name[(iy,ix)]=="POS":
                    self.well_positive_controls.append((iy,ix))

        self.well_positive_controls_number=len(self.well_positive_controls)

        # create a list of the drug names
        self.drug_names=(numpy.unique(self.well_drug_name)).tolist()


        self.drug_orientation={}
        for drug in self.drug_names:
            x=numpy.argwhere(self.well_drug_name==drug)
            n_rows=len(numpy.unique(x[:,0]))
            n_cols=len(numpy.unique(x[:,1]))
            if n_rows==1 and n_cols>1:
                self.drug_orientation[drug]='horizontal'
            elif n_rows>1 and n_cols==1:
                self.drug_orientation[drug]='vertical'
            elif n_rows>1 and n_cols==2:
                cols=numpy.unique(x[:,1])
                column0=(x[x[:,1]==cols[0]])
                column1=(x[x[:,1]==cols[1]])
                # look for the P-shape used for RIF on UKMYC6
                if column0[0][0]==column1[0][0] and column0[1][0]==column1[1][0]:
                    self.drug_orientation[drug]='P-shape'
                # otherwise the backwards L used for EMB on UKMYC5
                elif column0[0][-1]==column1[0][0]:
                    self.drug_orientation[drug]='L-shape'
                else:
                    self.drug_orientation[drug]='unknown'
            elif n_rows==1 and n_cols==2:
                self.drug_orientation[drug]='double-column'
            else:
                self.drug_orientation[drug]='unknown'

        if self.pixel_intensities:
            self.well_pixel_intensities={}
            for i in range(self.well_dimensions[0]):
                for j in range(self.well_dimensions[1]):
                    self.well_pixel_intensities[(i,j)]=[]

    def _scale_value(self,value):
        ''' Return a number scaled according to the size of the image.

        Most of the images used to develop this code have dimensions ca. 1000x660 since they were all captured using a Thermo Fisher Vizion instrument.
        This cannot be assumed, and therefore all relative coordinates have to be scaled according to the size of the image being processed. For example,
        if an image was 2000x1220 in size, then all linewidths and coordinate steps should be doubled.

        Args:
            value (int/float): a numerical value to adjust
            integer (bool=False): whether the numerical value is an integer, in which case the result will be rounded

        Returns:
            a number scaled according to the dimensions of the image
        '''
        if isinstance(value,int):
            return round(self.scaling_factor*value)
        else:
            return self.scaling_factor*value

    def save_arrays(self,file_ending):
        """ Save the numpy arrays with the well coordinates and growth etc in a single NPZ file.
        """

        # Ugly, but ensures that all the arrays keep their names when saved to the file.
        # Note that without compression the files are only 10Kb, which is << the size of the images
        # so there is no point compressing them. No difference in timing.
        numpy.savez(self.abspath+self.image_name+file_ending,
                                well_index=self.well_index,
                                well_radii=self.well_radii,
                                well_centre=self.well_centre,
                                well_top_left=self.well_top_left,
                                well_bottom_right=self.well_bottom_right,
                                well_growth=self.well_growth,
                                well_drug_name=self.well_drug_name,
                                well_drug_conc=self.well_drug_conc,
                                well_drug_dilution=self.well_drug_dilution,
                                threshold_pixel=self.threshold_pixel,
                                threshold_percentage=self.threshold_percentage,
                                sensitivity=self.sensitivity)

        if self.pixel_intensities:
            pickle.dump(self.well_pixel_intensities, open(self.abspath+self.image_name+"-pixels.pkl",'wb'))

    def load_arrays(self,file_ending,pixel_intensities=False):
        """ Load the numpy arrays with the well coordinates and growth etc from a single NPZ file.
        """

        # Also ugly, but at least it works and is explicit
        npzfile=numpy.load(self.abspath+self.image_name+file_ending)
        self.well_index = npzfile['well_index']
        self.well_radii = npzfile['well_radii']
        self.well_centre = npzfile['well_centre']
        self.well_top_left = npzfile['well_top_left']
        self.well_bottom_right = npzfile['well_bottom_right']
        self.well_growth = npzfile['well_growth']
        self.well_drug_name = npzfile['well_drug_name']
        self.well_drug_conc = npzfile['well_drug_conc']
        self.well_drug_dilution = npzfile['well_drug_dilution']
        self.threshold_pixel=npzfile['threshold_pixel']
        self.threshold_percentage=npzfile['threshold_percentage']
        self.sensitivity=npzfile['sensitivity']

        # create a list of the drug names
        self.drug_names=(numpy.unique(self.well_drug_name)).tolist()

        if pixel_intensities:
            self.well_pixel_intensities=pickle.load(open(self.abspath+self.image_name+"-pixels.pkl","rb"))

    def mean_shift_filter(self,spatial_radius=10,colour_radius=10):
        """ Apply a mean shift filter to produce a cleaner, more homogenous image.
        """

        # convert the 3 channel image to 1 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour=True

        # apply the mean shift filter
        self.image=cv2.pyrMeanShiftFiltering(self.image,spatial_radius,colour_radius)

    def equalise_histograms(self):
        """ Apply a global histogram filter.

        This performs much worse than equalise_histograms_locally() and is only included for comparision
        """

        # convert the 3 channel image to 1 channel
        if self.image_colour:
            self._convert_image_to_grey()
            self.image_colour=False

        # equalise the image histogram globally (will not take account of the uneven lighting)
        self.image=cv2.equalizeHist(self.image)

    def plot_histogram(self,file_ending):

        plt.tight_layout()
        fig = plt.figure(figsize=(4, 1.8))
        axis = plt.gca()
        axis.set_xlim([0,255])
        axis.axes.get_yaxis().set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        plt.hist(self.image.flatten(),bins=25,histtype='stepfilled',align='left',alpha=0.5,color="black",edgecolor='black',linewidth=1)
        plt.savefig(self.abspath+self.image_name+file_ending)

    def stretch_histogram(self,debug=False):

        mode=stats.mode(self.image,axis=None)[0]

        if debug:
            lower=numpy.percentile(self.image,5)
            upper=numpy.percentile(self.image,95)
            line="%s,%d,%d,%d" % (self.name,lower,mode,upper)

        self.image=(numpy.array(self.image,dtype=numpy.int16))-mode

        lower=numpy.percentile(self.image,5)
        upper=numpy.percentile(self.image,95)

        pos_factor=40./upper
        neg_factor=-110./lower

        self.image=numpy.multiply(self.image,numpy.where(self.image>0,pos_factor,neg_factor))
        self.image=self.image+180.

        lower=numpy.percentile(self.image,5)
        upper=numpy.percentile(self.image,95)
        mode=stats.mode(self.image,axis=None)[0]

        if debug:
            lower=numpy.percentile(self.image,5)
            upper=numpy.percentile(self.image,95)
            line+=",%d,%d,%d" % (lower,mode,upper)
            print(line)

    def equalise_histograms_locally(self):
        """ Apply a Contrast Limited Adaptive Histogram Equalization filter.

        Compared to a global histogram filter, this does a much better job of equalising the illumination across the plate.
        """

        # convert the 3 channel image to 1 channel
        if self.image_colour:
            self._convert_image_to_grey()
            self.image_colour=False

        # equalise the image histogram locally (will take account of the uneven lighting)
        # note that the tile grid matches the wells
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=self.well_dimensions)

        self.image = clahe.apply(self.image)

    def save_image(self,file_ending):
        """ Save the image to disc.
        """

        cv2.imwrite(self.abspath+self.image_name+file_ending,self.image)

    def _convert_image_to_colour(self):
        """ Convert the image to colour.
        """

        new_image=numpy.zeros(self.image.shape+(3,))
        for i in [0,1,2]:
            new_image[:,:,i]=self.image
        self.image=new_image

    def _convert_image_to_grey(self):
        """ Convert the image to greyscale
        """

        self.image=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def annotate_well_circumference(self,color=(0,0,0),linewidth=1):
        """ Draw circles around where AMyGDA has identified the wells on the plate.

        Args:
            color (B,G,R): Colour of the edge of the circle. Default is (0,0,0) which is black.
            linewidth (int): Width of the edge of the circle. Default is 1.
        """

        # convert the 1 channel image to 3 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour=True

        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):

                centre=(int(self.well_centre[(iy,ix)][0]),int(self.well_centre[(iy,ix)][1]))
                radius=int(self.well_radii[(iy,ix)])
                cv2.circle(self.image,centre,radius,color,linewidth)

    def annotate_well_centres(self,color=(0,0,0),linewidth=2):
        """ Draw a small circle at the centre of each well.

        Args:
            color (B,G,R): Colour of the edge of the circle. Default is (0,0,0) which is black.
            linewidth (int): Width of the edge of the circle. Default is 2.
        """

        # convert the 1 channel image to 3 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour=True

        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):

                centre=(int(self.well_centre[(iy,ix)][0]),int(self.well_centre[(iy,ix)][1]))
                radius=1
                cv2.circle(self.image,centre,radius,color,linewidth)

    def annotate_well_drugs_concs(self,color=(0,0,0),fontsize=0.4,fontemphasis=1):
        """ Label each well with the concentration of drug it contains.

        Args:
            color (B,G,R): Colour of the writing. Default is (0,0,0) which is black.
            fontsize (float): Relative size of the writing. Default is 0.4.
        """

        # convert the 1 channel image to 3 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour = True

        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):

                # label = "%02d" % self.well_index[(iy,ix)]
                label1 = "%s-%s" % (self.well_drug_name[(iy,ix)], self.well_drug_conc[(iy,ix)])
                label2 = "%s%%" % (int(self.well_growth[(iy,ix)]))

                (a,b) = (int(self.well_centre[(iy,ix)][0]),int(self.well_centre[(iy,ix)][1]))

                cv2.putText(self.image, label1, (int(a-self.well_radii[(iy,ix)]),b-round(20*self.scaling_factor)), cv2.FONT_HERSHEY_SIMPLEX, self.scaling_factor*fontsize, (255,0,0), fontemphasis)
                cv2.putText(self.image, label2, (a-round(15*self.scaling_factor),b+round(30*self.scaling_factor)), cv2.FONT_HERSHEY_SIMPLEX, self.scaling_factor*fontsize, (0,0,255), fontemphasis)

    def annotate_well_analysed_region(self,growth_color=(0,0,0),region=0.4,thickness=1): #,threshold_percentage=3,sensitivity=0):
        """ Draw coloured squares on the wells with detected bacterial growth.

        Args:
            growth_color (B,G,R): Colour of the square. Default is (0,0,0) which is black.
            region (float): The size of the square relative to the circle. Should match the value used in measure_growth. Default is 0.4.
            thickness (int): Width of the edge of the square. Default is 1.
        """

        # convert the 1 channel image to 3 channel
        if not self.image_colour:
            self._convert_image_to_colour()
            self.image_colour=True

        # check what threshold we should be using
        if self.sensitivity==0:
            growth_threshold_percentage=self.threshold_percentage
        elif (self.categories['IM_POS_AVERAGE']>(self.sensitivity*self.threshold_percentage)):
            growth_threshold_percentage=self.categories['IM_POS_AVERAGE']/self.sensitivity
        else:
            growth_threshold_percentage=self.threshold_percentage

        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):

                x=self.well_centre[(iy,ix)][0]
                y=self.well_centre[(iy,ix)][1]
                r=self.well_radii[(iy,ix)]*region

                if self.well_growth[iy,ix]>growth_threshold_percentage:
                    if self.categories["IM_POS_GROWTH"]:
                        cv2.circle(self.image,(x,y),int(r),growth_color,thickness=thickness)
                        # cv2.rectangle(self.image,(int(x-r),int(y-r)),(int(x+r),int(y+r)),growth_color,thickness=thickness)
                    else:
                        cv2.circle(self.image,(x,y),int(r),(0,0,0),thickness=thickness)
                        # cv2.rectangle(self.image,(int(x-r),int(y-r)),(int(x+r),int(y+r)),(0,0,0),thickness=thickness)

    def delete_mics(self):

        for drug in self.drug_names:
            self.categories.remove("IM_"+drug.upper()+"MIC")
            self.categories.remove("IM_"+drug.upper()+"DILUTION")

    def crop_based_on_hull(self, background_image, filename, hull):
        background_image_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
        mask = numpy.zeros(background_image_gray.shape, dtype='uint8')
        mask = cv2.drawContours(mask, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)
        cv2.imwrite(f"{filename}.mask.png", mask)
        result = []
        for background_image_pixel, mask_pixel in zip(background_image_gray.flatten(), mask.flatten()):
            if mask_pixel:
                result.append(background_image_pixel)
            else:
                result.append(255)
        result = numpy.array(result, dtype="uint8")
        result = result.reshape(background_image_gray.shape)
        cv2.imwrite(filename, result)

    def crop_based_on_hull_to_get_growth(self, image, background_image, filename, hull):
        background_image = cv2.bitwise_not(background_image)
        mask = numpy.zeros(image.shape, dtype='uint8')
        mask = cv2.drawContours(mask, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)
        cv2.imwrite(f"{filename}.mask.png", mask)
        # img2gray = cv2.bitwise_not(mask)
        # img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_and(background_image, background_image, mask=mask)
        result = cv2.bitwise_not(result)
        cv2.imwrite(filename, result)

        result_with_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result_with_color, [hull], -1, (0,255,0))
        cv2.imwrite(f"{filename}.with_contour.png", result_with_color)

        return result



    def get_countours(self, image):
        contours, hierarchies = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        inner_contours = []
        outer_contours = []
        for contour, hierarchy in zip(contours, hierarchies[0]):
            if hierarchy[-1] != -1:
                inner_contours.append(contour)
            else:
                outer_contours.append(contour)

        return contours, inner_contours, outer_contours

    def get_nth_largest_countour_and_its_area(self, contours, n):
        contours_area = [cv2.contourArea(contour) for contour in contours]
        contours_area = sorted(contours_area, reverse=True)
        nth_largest_contours_area = contours_area[n]
        nth_largest_contour = [contour for contour, contour_area in zip(contours, contours_area) if
                           cv2.contourArea(contour) == nth_largest_contours_area][0]
        return nth_largest_contour, nth_largest_contours_area



    def draw_contours(self, background_image, contours, filename, colors=((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        vis = background_image.copy()

        if len(vis.shape) < 3:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for index, contour in enumerate(contours):
            cv2.drawContours(vis, contour, -1, colors[index % len(colors)])
        cv2.imwrite(filename, vis)

    def get_hull_and_area_from_contour(self, contour):
        hull = cv2.convexHull(contour, False)
        hull_area = cv2.contourArea(hull)
        return hull, hull_area

    def get_center_of_contour(self, contour):
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy

    def remove_bubbles(self, image):
        return closing(image)

    def remove_condensation(self, image):
        # TODO
        return image

    def remove_sediments(self, image):
        # TODO
        return image

    def try_to_get_hull_with_no_shadows_and_outer_noise(self, second_largest_hull, area_of_the_second_largest_hull,
                                                        third_largest_hull, area_of_the_third_largest_hull, background_image, filename):
        big_difference_in_area = area_of_the_third_largest_hull < 0.4 * area_of_the_second_largest_hull

        if big_difference_in_area:
            appropriate_hull = second_largest_hull
            self.draw_contours(background_image, [[third_largest_hull], [second_largest_hull]],
                               f"{filename}.options_hulls.png")
        else:
            appropriate_hull = third_largest_hull
            self.draw_contours(background_image, [[second_largest_hull], [third_largest_hull]],
                               f"{filename}.options_hulls.png")

        return big_difference_in_area, appropriate_hull

    def get_distances_of_black_pixels_to_center_of_hull(self,appropriate_hull_image, x_center_of_appropriate_hull, y_center_of_appropriate_hull):
        distances_of_black_pixels_to_center_of_hull = []
        for y_coordinate in range(appropriate_hull_image.shape[0]):
            for x_coordinate in range(appropriate_hull_image.shape[1]):
                point_is_black = appropriate_hull_image[y_coordinate][x_coordinate] == 0
                if point_is_black:
                    distance = math.sqrt((x_coordinate - x_center_of_appropriate_hull)**2 +
                                         (y_coordinate - y_center_of_appropriate_hull)**2)
                    distances_of_black_pixels_to_center_of_hull.append(distance)
        return distances_of_black_pixels_to_center_of_hull

    def get_bins_for_histogram(self, distances):
        if len(distances) == 0:
            max_bin = 30
        else:
            max_bin = max(30, max(distances))
        return range(0, int(max_bin))

    def add_histogram_to_fig(self, ax, distances):
        ax.hist(distances, bins=self.get_bins_for_histogram(distances))
        loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)
        bottom, top = ax.get_ylim()  # return the current ylim
        top = max(top, 30)
        ax.set_ylim(0, top)

    def add_images_to_plot(self, fig, images_filenames):
        for index, image_filename in enumerate(images_filenames):
            image = cv2.imread(image_filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax = fig.add_axes([0.05 + 0.05 * index, 0.8, 0.2, 0.2])
            ax.imshow(image)
            ax.axis("off")

    def find_contours_and_return_appropriate_hull(self, image, background_image, filename):
        # transform bg image to BGR to draw with colours
        background_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)

        contours, inner_contours, outer_contours = self.get_countours(image)
        self.draw_contours(background_image, [contours], f"{filename}.full_countours.png")
        self.draw_contours(background_image, [inner_contours], f"{filename}.inner_contours.png")
        self.draw_contours(background_image, [outer_contours], f"{filename}.outer_contours.png")
        self.draw_contours(background_image, [inner_contours, outer_contours], f"{filename}.inner_outer_contours.png")

        second_largest_contour, area_of_the_second_largest_contour = \
            self.get_nth_largest_countour_and_its_area(contours, 1)
        third_largest_contour, area_of_the_third_largest_contour = \
            self.get_nth_largest_countour_and_its_area(contours, 2)
        self.draw_contours(background_image, [second_largest_contour, third_largest_contour], f"{filename}.second_and_third_largest_contour.png")

        second_largest_hull, area_of_the_second_largest_hull = \
            self.get_hull_and_area_from_contour(second_largest_contour)
        third_largest_hull, area_of_the_third_largest_hull = \
            self.get_hull_and_area_from_contour(third_largest_contour)
        self.draw_contours(background_image, [[second_largest_hull], [third_largest_hull]], f"{filename}.second_and_third_largest_hulls.png")

        # crop the background image based on the third_largest_hull
        self.crop_based_on_hull(background_image, f"{filename}.third_largest_hull_cropped.png", third_largest_hull)


        # get the good hull (with less noise)
        big_difference_in_area, appropriate_hull = self.try_to_get_hull_with_no_shadows_and_outer_noise(
            second_largest_hull,
            area_of_the_second_largest_hull,
            third_largest_hull,
            area_of_the_third_largest_hull,
            background_image,
            filename
        )
        self.draw_contours(background_image, [[appropriate_hull]], f"{filename}.chosen_hull.png")
        self.crop_based_on_hull(background_image, f"{filename}.chosen_hull.cropped.png", appropriate_hull)


        # we have now the following good hull
        chosen_hull_cropped_raw_image = cv2.imread(f'{filename}.chosen_hull.cropped.png')
        chosen_hull_cropped_raw_image = cv2.cvtColor(chosen_hull_cropped_raw_image,
                                                     cv2.COLOR_BGR2GRAY)

        # there is no noise of shadow or well border
        # but there could still have bubbles, condensation and sediment
        if self.apply_bubble_filter:
            chosen_hull_cropped_raw_image = self.remove_bubbles(chosen_hull_cropped_raw_image)
        cv2.imwrite(f'{filename}.chosen_hull.cropped.bubbles_removed.png', chosen_hull_cropped_raw_image)
        chosen_hull_cropped_raw_image = self.remove_condensation(chosen_hull_cropped_raw_image)
        chosen_hull_cropped_raw_image = self.remove_sediments(chosen_hull_cropped_raw_image)


        # Now the image is clean. Dark colours are growth
        # binarise with a fixed threshold
        _, chosen_hull_cropped_raw_image_binary_fixed = cv2.threshold(chosen_hull_cropped_raw_image, 120, 255,
                                                                             cv2.THRESH_BINARY)
        cv2.imwrite(f'{filename}.chosen_hull.cropped.binary_fixed.png', chosen_hull_cropped_raw_image_binary_fixed)

        appropriate_hull_image = chosen_hull_cropped_raw_image_binary_fixed

        good_choice = not big_difference_in_area

        return good_choice, appropriate_hull, appropriate_hull_image

    def get_area_of_ring(self, r1, r2):
        big_r = max(r1, r2)
        small_r = min(r1, r2)
        return (math.pi * (big_r**2)) - (math.pi * (small_r**2))

    def get_growth(self, appropriate_hull, appropriate_hull_image, initial_ring_radius=5, min_proportion_of_growth_initial_radius=0.01,
                   incremental_ring_radius=5, min_proportion_of_growth_incremental_radius=0.03):
        # TODO: add max_difference_in_growth_between_rings=0.3?
        x_center_of_appropriate_hull, y_center_of_appropriate_hull = self.get_center_of_contour(appropriate_hull)
        distances_of_black_pixels_to_center_of_hull = self.get_distances_of_black_pixels_to_center_of_hull(
            appropriate_hull_image, x_center_of_appropriate_hull, y_center_of_appropriate_hull
        )

        if len(distances_of_black_pixels_to_center_of_hull)==0:
            return 0.0

        max_distance = max(distances_of_black_pixels_to_center_of_hull)
        rings_radius = [0] + list(range(initial_ring_radius, int(max_distance) + incremental_ring_radius, incremental_ring_radius))
        histogram = numpy.histogram(distances_of_black_pixels_to_center_of_hull, rings_radius)
        nb_black_pixels_in_each_bin = numpy.array(histogram[0])
        rings_area =  numpy.array([ self.get_area_of_ring(small_ring, big_ring) for small_ring, big_ring in zip(rings_radius[:-1], rings_radius[1:])])
        density_per_ring = nb_black_pixels_in_each_bin / rings_area
        min_proportion_of_growth = [min_proportion_of_growth_initial_radius] + [min_proportion_of_growth_incremental_radius]*(len(density_per_ring)-1)
        rings_indexes_with_too_low_growth = numpy.argwhere(density_per_ring < min_proportion_of_growth)
        if len(rings_indexes_with_too_low_growth)==0:
            ring_limit = len(nb_black_pixels_in_each_bin)
        else:
            ring_limit = rings_indexes_with_too_low_growth[0][0]

        nb_of_black_pixels_in_area_we_are_confident_it_is_growth = numpy.sum(nb_black_pixels_in_each_bin[:ring_limit])
        area_of_appropriate_hull = cv2.contourArea(appropriate_hull)
        growth = nb_of_black_pixels_in_area_we_are_confident_it_is_growth / area_of_appropriate_hull * 100

        print(f"histogram: {histogram}")
        print(f"nb_black_pixels_in_each_bin: {nb_black_pixels_in_each_bin}")
        print(f"density_per_ring: {density_per_ring}")
        print(f"ring_limit: {ring_limit}")
        print(f"nb_of_black_pixels_in_area_we_are_confident_it_is_growth: {nb_of_black_pixels_in_area_we_are_confident_it_is_growth}")
        print(f"area_of_appropriate_hull: {area_of_appropriate_hull}")
        print(f"growth: {growth}")

        return growth


    def plot(self, filename, appropriate_hull, appropriate_hull_image):
        x_center_of_appropriate_hull, y_center_of_appropriate_hull = self.get_center_of_contour(appropriate_hull)
        distances_of_black_pixels_to_center_of_hull = self.get_distances_of_black_pixels_to_center_of_hull(
            appropriate_hull_image, x_center_of_appropriate_hull, y_center_of_appropriate_hull
        )

        fig, ax = plt.subplots(figsize=(20, 5))
        self.add_histogram_to_fig(ax, distances_of_black_pixels_to_center_of_hull)
        self.add_images_to_plot(fig,
                                [
                                    f'{filename.replace("_cropped", ".raw.padded.png")}',
                                    f'{filename.replace("_cropped", ".second_and_third_largest_hulls.png")}',
                                    f'{filename.replace("_cropped", ".third_largest_hull_cropped.png")}',
                                    f'{filename}.options_hulls.png',
                                    f'{filename}.chosen_hull.cropped.png',
                                    f'{filename}.chosen_hull.cropped.bubbles_removed.png',
                                    f'{filename}.chosen_hull.cropped.binary_fixed.png',
                                ])

        plt.savefig(f'{filename}.histogram.png')
        plt.close()

        self.histogram_files.append(f'{filename}.histogram.png')

    @staticmethod
    def pad_image(image):
        top = int(0.2 * image.shape[0])
        bottom = top
        left = int(0.2 * image.shape[1])
        right = left
        dst = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 255)
        return dst


    def output_final_report(self):
        doc, tag, text = Doc().tagtext()
        report_path = self.abspath+"html_report/"
        os.makedirs(report_path, exist_ok=True)
        images_path = self.abspath + "html_report/images/"
        os.makedirs(images_path, exist_ok=True)

        with tag('html'):
            with tag('body'):
                with tag('div'):
                    with tag('h1'):
                        text("Raw plate image:")
                    doc.stag('img', src=f"images/{os.path.basename(self.image_path)}")
                    shutil.copy(self.image_path, images_path)

                with tag('div'):
                    with tag('h1'):
                        text("Analysis:")
                    with tag('h3'):
                        text("Some explanation:")
                    with tag('p'):
                        text("In what follows we have a histogram of the growth for each well.")
                        doc.stag("br")
                        text("In each plot we have 3 images. The first is the raw well image;")
                        doc.stag("br")
                        text("The second includes a green region that we selected to analyse the growth (we try to get a region inside the well that has less noise);")
                        doc.stag("br")
                        text("The third image is the actual image we process: a binarised image of the highlighted region in the second image.")
                        doc.stag("br")
                        text("The growth is measured as black pixels in the 3rd image;")
                        doc.stag("br")
                        text("The histogram denotes the number of black pixels (growth - y-axis) by the distance of the black pixels to the centroid of the region (x-axis);")
                        doc.stag("br")
                        text("The idea is that we still have noise, and they are usually on the border of the region. So if the histogram looks like the growth *just* appears in the border, that is usually noise, and we should cut it off;")
                        doc.stag("br")
                        text("Otherwise, if the histogram shows steady growth from some point in the beginning until some more distant point, we know it is growth;")
                        doc.stag("br")
                        text("If the growth is in the whole cell, then we can consider also that the black points in the border are not noise, but growth;")
                        doc.stag("br")
                        text("TODO: We still need to define how to cut the histogram;")
                        doc.stag("br")
                        text("TODO: It seems we do well in cases where we have sediments and shadows, but we don't do well with air bubbles, condensation and plate failure yet (see Fig 3 of https://www.biorxiv.org/content/10.1101/229427v4.full.pdf);")
                        doc.stag("br")
                        text("TODO: Some way to flag dodgy wells e.g. worm well;")
                        doc.stag("br")
                        text("TODO: scale/normalise growth in chosen area for well size;")

                with tag('h1'):
                    text("Observations: ")

                with tag('h3'):
                    text(f"{'THERE IS' if self.there_is_growth_dispersion_in_control_well else 'THERE IS NOT'} dispersed growth in control wells,"
                         f"so we are {'ENABLING' if self.apply_bubble_filter else 'DISABLING'} the bubble filter.")

                with tag('h1'):
                    text("Wells:")
                for index, histogram_file in enumerate(self.histogram_files):
                    row = int(index / self.well_dimensions[1])
                    column = index % self.well_dimensions[1]
                    with tag('h1'):
                        text(f'Well {row} {column}:')
                    doc.stag("br")
                    doc.stag('img', src=f"images/{os.path.basename(histogram_file)}")
                    shutil.copy(histogram_file, images_path)

        result = doc.getvalue()
        with open(report_path+"report.html", "w") as fout:
            fout.write(result)


    def process_well(self, iy, ix, c_param, zoom_out_factor=0.1, add_to_report=False):
        try:
            x = self.well_centre[(iy, ix)][0]
            y = self.well_centre[(iy, ix)][1]
            r = self.well_radii[(iy, ix)] * (1.0 + zoom_out_factor)
            max_y = self.image.shape[0]
            max_x = self.image.shape[1]

            # get the original raw image of the well only
            original_raw_image = self.image[max(0, int(y - r)):min(int(y + r), max_y),
                                 max(0, int(x - r)):min(int(x + r), max_x)]
            cv2.imwrite(f"{self.debug_images}well_{iy}_{ix}.raw.png", original_raw_image)

            # pad the image
            padded_raw_image = self.pad_image(original_raw_image)
            cv2.imwrite(f"{self.debug_images}well_{iy}_{ix}.raw.padded.png", padded_raw_image)

            # recalculate and draw the circle in the image
            padded_raw_image_with_circles = padded_raw_image.copy()
            # TODO: take a look at these cv2.HoughCircles() params
            circles = cv2.HoughCircles(padded_raw_image_with_circles, cv2.HOUGH_GRADIENT, 1, 50, param1=20,
                                       param2=25, minRadius=int(0.6 * r), maxRadius=int(1.2 * r))
            if circles is None: circles = []
            if len(circles) != 1:
                print(f"{len(circles)} circles found for well {iy} {ix}, should be 1")
                assert False
            circle_x, circle_y, circle_radius = circles[0][0]
            cv2.circle(padded_raw_image_with_circles, (int(circle_x), int(circle_y)), int(circle_radius), 0, thickness=2)
            cv2.imwrite(f"{self.debug_images}well_{iy}_{ix}.raw.padded_with_circles.png", padded_raw_image_with_circles)

            # apply adaptive threshold
            # TODO: this block size is too large I think... it makes very thick contours
            # TODO: if needed to close up some circles, we should be looking at https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.erosion
            # TODO: but this could enlarge growth as well
            block_size = int(r)
            if block_size % 2 == 0: block_size += 1
            binary_image = cv2.adaptiveThreshold(padded_raw_image_with_circles, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY,
                                                 block_size,
                                                 c_param)
            cv2.imwrite(f"{self.debug_images}well_{iy}_{ix}.binary.png", binary_image)

            # find the hull inside the well
            self.find_contours_and_return_appropriate_hull(binary_image,
                                                           padded_raw_image,
                                                           f"{self.debug_images}well_{iy}_{ix}")

            # find a even less noisy hull inside the well
            for variable_c_param in range(c_param, 0, -1):
                # TODO: I do think we should reduce the block size here
                # for block_size in range(5, block_size+1, 4):
                    cropped_image = cv2.imread(f"{self.debug_images}well_{iy}_{ix}.third_largest_hull_cropped.png")
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    cropped_binary_image = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                 cv2.THRESH_BINARY,
                                                                 block_size,
                                                                 variable_c_param)

                    cv2.imwrite(f"{self.debug_images}well_{iy}_{ix}_cropped.raw.png", cropped_image)
                    cv2.imwrite(f"{self.debug_images}well_{iy}_{ix}_cropped.binary.png", cropped_binary_image)
                    good_choice, appropriate_hull, appropriate_hull_image = self.find_contours_and_return_appropriate_hull(cropped_binary_image,
                                                                                                   padded_raw_image,
                                                                                                   f"{self.debug_images}well_{iy}_{ix}_cropped")

                    if good_choice:
                        break

            if good_choice == False:
                # we never got a good choice, fallback to the first one
                cropped_image = cv2.imread(f"{self.debug_images}well_{iy}_{ix}.third_largest_hull_cropped.png")
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                cropped_binary_image = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                             cv2.THRESH_BINARY,
                                                             block_size,
                                                             c_param)

                cv2.imwrite(f"{self.debug_images}well_{iy}_{ix}_cropped.raw.png", cropped_image)
                cv2.imwrite(f"{self.debug_images}well_{iy}_{ix}_cropped.binary.png", cropped_binary_image)
                good_choice, appropriate_hull, appropriate_hull_image = self.find_contours_and_return_appropriate_hull(cropped_binary_image,
                                                                                               padded_raw_image,
                                                                                               f"{self.debug_images}well_{iy}_{ix}_cropped")
            if add_to_report:
                self.plot(f"{self.debug_images}well_{iy}_{ix}_cropped", appropriate_hull, appropriate_hull_image)

            print(f"well {iy} {ix}")
            self.well_growth[iy, ix] = self.get_growth(appropriate_hull, appropriate_hull_image)
        except: # TODO: get errors and process/fix them correctly
            self.well_growth[iy, ix] = -1.0

    def get_nb_black_pixels_in_image(self, image):
        return numpy.count_nonzero(image == 0)


    def infer_if_there_is_growth_dispersion_in_control_well(self, y, x):
        # TODO: here we just infer if there are 5+ medium or large growth contours, this should be improved
        unfiltered_image = cv2.imread(f"{self.debug_images}well_{y}_{x}_cropped.chosen_hull.cropped.png")
        unfiltered_image = cv2.cvtColor(unfiltered_image, cv2.COLOR_BGR2GRAY)
        _, unfiltered_image_binary_fixed = cv2.threshold(unfiltered_image, 120, 255, cv2.THRESH_BINARY)

        contours, _, _ = self.get_countours(unfiltered_image_binary_fixed)

        contours_area = sorted([cv2.contourArea(contour) for contour in contours], reverse=True)
        contours_area = contours_area[1:] # remove the outer contour area

        # remove very small growth (noise)
        contours_area = [contour_area for contour_area in contours_area if contour_area >= 10]

        there_are_five_or_more_non_small_growth = len(contours_area) >= 5

        return there_are_five_or_more_non_small_growth

        # Previous tries for dispersion detection:
        # # compute index of dispersion (https://en.wikipedia.org/wiki/Index_of_dispersion)
        # variance = numpy.var(contours_area)
        # mean = numpy.mean(contours_area)
        # index_of_dispersion = variance**2 / mean
        # print(index_of_dispersion)
        # return True

        # there_is_a_big_contour = any([contour_area > 50 for contour_area in contours_area])
        # number_of_small_contours = sum([contour_area < 50 for contour_area in contours_area])
        # proportion_of_small_contours = number_of_small_contours / len(contours_area)
        # return number_of_small_contours >= 5 and proportion_of_small_contours >= 0.8





    def infer_if_we_have_growth_dispersion_in_control_wells(self, c_param):
        self.process_well(7, 10, c_param)
        there_is_growth_dispersion_in_control_well_1 = self.infer_if_there_is_growth_dispersion_in_control_well(7, 10)

        self.process_well(7, 11, c_param)
        there_is_growth_dispersion_in_control_well_2 = self.infer_if_there_is_growth_dispersion_in_control_well(
            7, 11)

        return there_is_growth_dispersion_in_control_well_1 or there_is_growth_dispersion_in_control_well_2



    def measure_growth(self,threshold_pixel=120,threshold_percentage=3,region=0.4,sensitivity=4.0, c_param = 10):
        """ Analyse each of the wells and decide if there is bacterial growth.

        This is all based on the pixels found in a central square region.
        Note: the pixels have intensities (0-255) where 0 is dark and 255 is light.

        Args:
            threshold_pixel (int): pixels with intensities above this value are assumed to be bacterial growth_threshold_percentage.
                Default is 130. Range 0 to 255.
            threshold_percentage (float): the proportion of the central region of each well that contains bacterial growth for
                the well to be classifed as containing bacterial growth
            region (float): the size of the central square relative to the estimated size of the well. Default is 0.4.
            sensitivity (float): if the average percentage growth in the control wells is greater than this value multiplied by
                threshold_percentage, then, rather than threshold_pixel, use the average percentage growth in the control wells divided by
                sensitivity. This is a higher threshold and better deals with artefacts. The default is 4.0.
                E.g. using defaults, for a plate with 20% average growth in the control wells a threshold of 5%, rather than 3%, would be applied.
        """

        # check the parameters lie within the required ranges
        assert 255>=threshold_pixel>=0, "threshold_pixel must take a value between 0 and 255"
        assert 100>=threshold_percentage>=0, "threshold_percentage must take a value between 0 and 100"
        assert 1>=region>=0, "if region is larger than 1 it will include parts of the well circumference."

        # convert the 3 channel image to 1 channel
        if self.image_colour:
            self._convert_image_to_grey()
            self.image_colour=False

        self.threshold_pixel=threshold_pixel
        self.threshold_percentage=threshold_percentage
        self.sensitivity=sensitivity

        self.there_is_growth_dispersion_in_control_well = self.infer_if_we_have_growth_dispersion_in_control_wells(c_param)
        self.apply_bubble_filter = not self.there_is_growth_dispersion_in_control_well


        for iy in range(0,self.well_dimensions[0]):
            for ix in range(0,self.well_dimensions[1]):
                self.process_well(iy, ix, c_param, add_to_report=True)

        # print(self.well_growth)
        # self.output_final_report()

        counter=1
        positive_control_growth_total=0.0

        # start off assuming there is growth in the control wells
        self.categories['IM_POS_GROWTH']=True

        # loop through the control wells (any well with no conc of a drug)
        for control_well in self.well_positive_controls:

            positive_control_growth=self.well_growth[control_well]

            positive_control_growth_total+=positive_control_growth

            # store the growth in the control well as a % to 2DP
            self.categories['IM_POS'+str(counter)+'GROWTH']=float("%.2f" % (positive_control_growth))

            if positive_control_growth>self.threshold_percentage:
                self.categories['IM_POS'+str(counter)]=True
            else:
                self.categories['IM_POS'+str(counter)]=False
                # if there is isn't growth a single control well set this to False
                self.categories['IM_POS_GROWTH']=False

            counter+=1

        # calculate the average growth (%) in the control wells, limited to 2DP
        self.categories['IM_POS_AVERAGE']=float("%.2f" % (positive_control_growth_total/self.well_positive_controls_number))

        # reset all the image processing fields
        # for drug in self.drug_names:
        #     self.categories["IM_"+drug.upper()+"MIC"]=None
        #     self.categories["IM_"+drug.upper()+"DILUTION"]=None

        number_drugs_inconsistent_growth=0

        # self.categories["IM_DRUGS_INCONSISTENT_GROWTH"]=None

        # check what threshold we should be using
        if self.sensitivity==0:
            growth_threshold_percentage=self.threshold_percentage
        elif (self.categories['IM_POS_AVERAGE']>(self.sensitivity*self.threshold_percentage)):
            growth_threshold_percentage=self.categories['IM_POS_AVERAGE']/self.sensitivity
        else:
            growth_threshold_percentage=self.threshold_percentage

        # now iterate through the drugs
        for drug in self.drug_names:

            # remember to reverse the order so we go from low to high conc
            growth=self.well_growth[self.well_drug_name==drug][::-1]
            conc=self.well_drug_conc[self.well_drug_name==drug][::-1]
            dilution=self.well_drug_dilution[self.well_drug_name==drug][::-1]

            # this is a bit hacky, but the problem was the above arrays were not
            # being returned sorted for EMB since it "goes around the corner"
            # this logic uses dilution to correctly order the wells, then resets it
            growth=growth[dilution-1]
            conc=conc[dilution-1]
            dilution=numpy.arange(len(dilution))+1

            # start off with no MIC and not having seen anything
            mic_conc=None
            mic_dilution=None
            seen_growth=False
            seen_no_growth=False
            inconsistent_growth=False

            # iterate through the strip of wells
            for (g,c,d) in zip(growth,conc,dilution):

                # ..otherwise no choice but to apply a simple threshold
                if g>growth_threshold_percentage:

                    seen_growth=True

                    # but if we have already seen no growth
                    if seen_no_growth:
                        inconsistent_growth=True

                # ..otherwise there is no growth
                else:

                    # and we haven't set an MIC yet
                    if not mic_conc:

                        # ..but we have seen growth and not yet seen no growth
                        if seen_growth and not seen_no_growth:

                            # set the MIC values
                            mic_conc=c
                            mic_dilution=d

                    # and now remember that we've seen a well with no growth in it
                    seen_no_growth=True

            # deal with the case where there is growth in ALL wells
            #  (cannot infer that the concentration of the next doubling would be sufficient to kill the bug)
            #  (hence have to report as anomalous/no result)
            if seen_growth and not seen_no_growth and not mic_conc:
                mic_conc=numpy.max(conc)*2
                mic_dilution=numpy.max(dilution)+1

            # deal with the case where there is no growth in ALL wells
            #  (can infer that the concentration of the first well is at least an upper limit for the MIC)
            if seen_no_growth and not seen_growth and not mic_conc:
                mic_conc=numpy.min(conc)
                mic_dilution=1

            # what if there is anomalous growth?
            if inconsistent_growth:
                mic_conc=-1
                mic_dilution=-1
                number_drugs_inconsistent_growth+=1

            # what if there is no growth in the control wells?
            if self.categories['IM_POS_GROWTH']==False:
                mic_conc=-2
                mic_dilution=-2

            # translate into text for storing in the treant
            if mic_conc is not None:
            #     self.categories["IM_"+drug.upper()+"MIC"]=None
            #     self.categories["IM_"+drug.upper()+"DILUTION"]=None
            # else:
                self.categories["IM_"+drug.upper()+"MIC"]=float(mic_conc)
                self.categories["IM_"+drug.upper()+"DILUTION"]=int(mic_dilution)


        assert number_drugs_inconsistent_growth>=0
        self.categories["IM_DRUGS_INCONSISTENT_GROWTH"]=number_drugs_inconsistent_growth

    def write_mics(self,file_ending):
        """ Write the results to a plaintext file.

        These are simply all the data stored in the JSON file, but this is difficult to read, so this
        function provides a simple way to produce human-readable files. It will include the MICs and dilutions
        for each well, as well as information about the growth in the control well(s).
        """

        OUTPUT = open(self.abspath+self.image_name+file_ending,'w')

        for field in sorted(self.categories.keys()):
            OUTPUT.write("%28s %20s\n" % (field, self.categories[field]))

        OUTPUT.close()

    def identify_wells(self,hough_param1=20,hough_param2=25,radius_tolerance=0.005,verbose=False):
        """ Using a Hough transform identify the wells.

        Only if the number of circles found is the same as the number of wells is True returned.

        Args:
            radius_tolerance (float): the rate at which the min/max radii are decreased/increased. Increasing it speeds up
                the search, but makes it more likely it will 'miss' identifying the correct number of wells. default=0.005
            hough_param1 (int) and hough_param2 (int): for an explanation of these, please see the OpenCV documentation
                e.g. https://docs.opencv.org/3.1.0/d4/d70/tutorial_hough_circle.html
                e.g. https://dsp.stackexchange.com/questions/22648/in-opecv-function-hough-circles-how-does-parameter-1-and-2-affect-circle-detecti
        """

        # estimate the dimensions of the well
        estimate_well_y = float(self.image_dimensions[0])/self.well_dimensions[0]
        estimate_well_x = float(self.image_dimensions[1])/self.well_dimensions[1]

        # verify that the estimated dimensions of the wells are within 5% of one another
        if estimate_well_x > estimate_well_y:
            if estimate_well_x > 1.05*estimate_well_y:
                return False
        else:
            if estimate_well_y > 1.05*estimate_well_x:
                return False

        # now estimate the radius as the mean of half the estimated well dimension
        estimated_radius=(estimate_well_x + estimate_well_y)/4.

        radius_multiplier=1.+radius_tolerance

        # read in a greyscale image for identifying
        grey_image = cv2.imread(self.image_path,0)

        # iterate until the correct number of circles have been found
        while True:

            circles=None

            while circles is None:

                # detect circles using the Hough transform
                circles=cv2.HoughCircles(grey_image,cv2.HOUGH_GRADIENT,1,50,param1=hough_param1,param2=hough_param2,minRadius=int(estimated_radius/radius_multiplier),maxRadius=int(estimated_radius*radius_multiplier))

                # increase the range of radii that will be searched for
                radius_multiplier+=radius_tolerance

                if verbose:
                    print("No circles, "+str(int(estimated_radius/radius_multiplier))+" < radius < "+str(int(estimated_radius*radius_multiplier)))

            # count how many circles there are
            number_of_circles=len(circles[0])

            if verbose:
                print(str(number_of_circles)+" circles, "+str(int(estimated_radius/radius_multiplier))+" < radius < "+str(int(estimated_radius*radius_multiplier)))

            # check to see if there are enough circles
            if number_of_circles>=self.number_of_wells:
                break
            elif number_of_circles>self.number_of_wells:
                # print(self.image_path+": "+str(number_of_circles)+" circles found, this is more than the expected number of "+str(self.number_of_wells))
                break
            elif radius_multiplier>2:
                # print(self.image_path+": "+"reached maximum radius multiplier of 2 and found "+str(number_of_circles)+" circles, giving up")
                break
            else:
                radius_multiplier+=radius_tolerance

        well_counter=0

        one_circle_per_well=True

        # move along the wells in the x-direction (columns)
        for ix in range(0,self.well_dimensions[1]):

            # move along the wells in the y-direction (rows)
            for iy in range(0,self.well_dimensions[0]):

                # calculate the bottom left and top right coordinates of the well
                top_left=(int(ix*estimate_well_x), int(iy*estimate_well_y))
                bottom_right=(int((ix+1)*estimate_well_x), int((iy+1)*estimate_well_y))

                number_of_circles_in_well=0

                # loop over the list of identified wells
                for ic in circles[0,]:

                    # check to see if the circle lies with the well
                    if top_left[0] < ic[0] < bottom_right[0]:
                        if top_left[1] < ic[1] < bottom_right[1]:
                            number_of_circles_in_well+=1
                            circle=ic

                if number_of_circles_in_well==1:

                    well_centre=(circle[0],circle[1])
                    well_radius=circle[2]
                    well_extent=1.2*well_radius

                    x1=max(0,int(well_centre[0]-well_extent))
                    x2=min(self.image_dimensions[1],int(well_centre[0]+well_extent))

                    y1=max(0,int(well_centre[1]-well_extent))
                    y2=min(self.image_dimensions[0],int(well_centre[1]+well_extent))

                    self.well_index[iy,ix]=well_counter
                    self.well_centre[iy,ix] = well_centre
                    self.well_radii[iy,ix] = well_radius
                    self.well_top_left[iy,ix] = (x1,y1)
                    self.well_bottom_right[iy,ix] = (x2,y2)
                    well_counter+=1
                else:
                    if verbose:
                        print(number_of_circles_in_well)
                    one_circle_per_well=False



        if well_counter==self.number_of_wells and one_circle_per_well:
            return True
        else:
            return False
