### Opening an image, selecting objects, creating and base filtering of cell objects
from skimage.feature import blob_dog, blob_log, blob_doh
import matplotlib.pyplot as plt
import numpy as np
import skimage
import imageio
from skimage.morphology import erosion, dilation, opening, closing, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage import measure
from scipy import ndimage
from skimage.draw import polygon_perimeter
import pickle
import tifffile
import skimage.io
import skimage.color
import skimage.filters
from cellobj import CellObj, add_measured_value, save_objects_as_pickle

def thresholder(image, size=1.0, background='black', channel='all'):
    '''thresholds an image with otsu method, filters noise'''
    # add median filter as an option latter
    if channel != 'all':
        image = image[...,channel]
        image = np.stack((image,)*3, axis=-1)
    image_gs = skimage.color.rgb2gray(image)
    image_blured = skimage.filters.gaussian(image_gs, sigma=size)
    threshold = skimage.filters.threshold_otsu(image_blured)
    mask =  image_blured > threshold
    if background != 'black':
        mask = np.invert(mask)

    # add fill holes and morphology operations
    mask = ndimage.binary_fill_holes(mask)
    return mask

def detector(image, margin, constant_value=0.8):
    ''' detects particles using contours. limits bounding boxes woith the size of image'''
    contours = measure.find_contours(image, constant_value)
    bounding_boxes = []
    for contour in contours:
        #prevent values which are negative or outside the image
        neg_limiter = lambda x : 0 if x < 0 else x
        pos_limiter_x = lambda x : image.shape[0] if x > image.shape[0] else x
        pos_limiter_y = lambda x : image.shape[1] if x > image.shape[1] else x

        Ymin = neg_limiter(np.min(contour[:,0]).astype(int) - margin)
        Ymax = pos_limiter_x(np.max(contour[:,0]).astype(int) + margin)
        Xmin = neg_limiter(np.min(contour[:,1]).astype(int) - margin)
        Xmax = pos_limiter_y(np.max(contour[:,1]).astype(int) + margin)

        bounding_boxes.append([Ymin, Ymax, Xmin, Xmax])

    return contours, bounding_boxes

def object_bounding_boxes(image,detector, margin=0):
    '''detects particles of a given size, returns its coordinates in a list. Accepts custom detector, image or thresholded image'''
    contours,bounding_boxes  = detector(image, margin)
    return (contours, bounding_boxes)

def create_cell_list(box_coord_list, image, mask, image_hash, contours = None):
    '''iterates over coordinate lists, create cell objects, store them into a list, return list of the given objects'''
    cell_obj_list = []
    for count, box_coord in enumerate(box_coord_list):
        if contours:
            contour = contours[count]
        else:
            contour = None
        cell_obj_list.append(CellObj(count, box_coord, image, mask, contour, image_hash))

    return cell_obj_list


def filter_cell_objects_by_size(cell_list, image, show_hist=False, quantile=0.05):
# Drawing histogramm and calculating values lower than 0.05 quantile for cell size filtering
    def is_cell_bordering(cell_obj, image):
        '''returns True if CellObj box borders the margin of an image'''
        if (((cell_obj.box_coord[0] == 0) or (cell_obj.box_coord[2] == 0)) or
            (cell_obj.box_coord[1] >= (image.shape[0]-1) or
            (cell_obj.box_coord[3] >= (image.shape[1]-1)))):
            return True
        else:
            return False

    cell_area = []
    for cell in cell_list:
        cell_area.append(cell.return_area())
    if show_hist:
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 5)
        ax.hist(cell_area , bins=40)
        plt.title(f'{quantile} quantile: {str(round(np.quantile(cell_area, 0.05),2))}')
        ax.axvline(x=np.quantile(cell_area, 0.05), color ='red')
        plt.show

    cell_min_size = np.quantile(cell_area, quantile)

    cell_list_cleared = []
    for cell in cell_list:
        if (is_cell_bordering(cell, image) == False) and (cell.return_area() > cell_min_size):
            cell_list_cleared.append(cell)
    return cell_list_cleared

def show_contours_and_boxes_on_image(image, cell_obj_list):
    '''from https://muthu.co/draw-bounding-box-around-contours-skimage/
    Draws a bounding box and contour if present around objects'''
    bounding_boxes = [cell.box_coord for cell in cell_obj_list]
    contours = [cell.contour for cell in cell_obj_list]
    boxed_img = np.copy(image)
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    if contours:
        for n, contour in enumerate(contours):
            ax.plot(contours[n][:, 1], contours[n][:, 0], linewidth=2)

    for box in bounding_boxes:
        r = [box[0],box[1],box[1],box[0], box[0]]
        c = [box[3],box[3],box[2],box[2], box[3]]
        rr, cc = polygon_perimeter(r, c, image.shape)
        boxed_img[rr, cc] = (255,255,255) #set color white
    ax.imshow(boxed_img, interpolation='nearest', cmap=plt.cm.gray)
    plt.grid(False)
    plt.axis('off')
    plt.show()
