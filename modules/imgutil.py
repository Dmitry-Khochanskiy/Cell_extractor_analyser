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
from modules.cellobj import CellObj, add_measured_value, save_objects_as_pickle


def image_loader(path):
    '''loads an image, converts it into pil'''
    image = skimage.io.imread(path)
    return image, str(hash(tuple(image.tobytes())))

def show_image(image, size=(18.5, 10.5), title=""):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax.grid(False)
    plt.show()

def show_cell_and_mask(cell_obj):
    '''shows induvidual cell from a cell object'''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cell_obj.images_dict['cut_image'], interpolation='nearest', cmap=plt.cm.gray)
    ax2.imshow(cell_obj.images_dict['mask_image'], interpolation='nearest', cmap=plt.cm.gray)
    plt.show()

def save_objects_as_pickle(cell_list, path):
    with open(f'{path}/cell_objects.pickle', 'wb') as handle:
        pickle.dump(cell_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_objects_as_tiff(cell_list, path, n_channels=3):
    # finds the biggest bounding box
    largest_box = [0, 0]
    for cell in cell_list:
        x_dim = cell.box_coord[1] - cell.box_coord[0]
        y_dim = cell.box_coord[3] - cell.box_coord[2]
        if x_dim > largest_box[0]:
            largest_box[0] = x_dim
        if y_dim > largest_box[1]:
            largest_box[1] = y_dim
    #creates an empty stack with this size
    stack = np.zeros((largest_box[0],largest_box[1]))
    #add automatic dimension increase for RGB or other nDimensions
    stack = np.stack((stack,)*n_channels, axis=-1)
    stack = np.stack((stack,), axis=0)
    #reshaping each cell array and adding to the stack
    for cell in cell_list:
        array = cell.images_dict['cut_image']
        bigger_array = np.zeros((largest_box[0],largest_box[1]))
        bigger_array.fill(255)
        bigger_array  = np.stack((bigger_array,)*n_channels, axis=-1)
        x = 0
        y = 0
        bigger_array[x:x+array.shape[0], y:y+array.shape[1]] = array
        bigger_array = np.expand_dims(bigger_array,0)
        stack = np.concatenate((stack,bigger_array),axis=0)
    tifffile.imwrite(f'{path}/stack.tiff', stack[1:,...], photometric='rgb')
