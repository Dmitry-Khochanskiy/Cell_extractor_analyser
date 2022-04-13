### Calculating various paramters for cell object, storing them into CSV files and objects themselves
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.ndimage import center_of_mass
from scipy.signal import correlate
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.measure import label, regionprops, regionprops_table
from scipy import ndimage
import pickle
import tifffile
import pandas as pd
import skimage.io
import skimage.color
import skimage.filters
from scipy.stats import pearsonr
from modules.cellobj import CellObj, add_measured_value, save_objects_as_pickle


def add_measured_value(cell_obj_list, function, **kwargs):
    '''Adds calculated measurments to every cell object in a list usong functions'''
    new_cell_obj_list = []

    for cell_obj in cell_obj_list:
        props = function(cell_obj, **kwargs)
        # accepts return as a dictionary, iterates over dictionary and adds it to value dictionary
        for key, value in props.items():
            cell_obj.measured_values[key] = value
        new_cell_obj_list.append(cell_obj)

    return new_cell_obj_list

def calculate_morpho_properties(cell_obj, properties):
    '''returns a dict of a given moprhological measurments from skimage.measure'''
    props = regionprops_table(cell_obj.images_dict['mask'].astype('int'), properties=properties)
    return_dict = {}
    for key, value in props.items():
        return_dict[key] = float(value)
    return return_dict

def calculate_intensity(cell_obj, use_mask=False):
    # Can be used as a baseline for other functions
    '''calculate intensity and std for all channels based on mask '''
    image = cell_obj.images_dict['cut_image']
    #if there is no mask the whole image is masked
    if use_mask:
        mask = cell_obj.images_dict['mask']
        mask = np.invert(mask)
    else:
        mask = np.ones_like(image).astype(bool)
        mask = np.invert(mask)

    mean_std_dict = {}
    return_dict = {}

    masked_image = np.ma.array(image, mask = mask)
    image_mean = round(np.mean(masked_image),2)
    image_std = round(np.std(masked_image),2)

    mean_std_dict['mean_intensity'] = image_mean
    mean_std_dict['mean_std'] = image_std

    for key, value in mean_std_dict.items():
        return_dict[key] = float(value)
    return return_dict

def calculate_intensity_per_channel(cell_obj, use_mask=False):
    # Rewrite or ditch
    '''calculate intensity and std for each channel based on mask '''
    image = cell_obj.images_dict['cut_image']
    #if there is no mask the whole image is masked
    if use_mask:
        mask = cell_obj.images_dict['mask']
        mask = np.invert(mask)
    else:
        mask = np.ones_like(image[...,0]).astype(bool)
        mask = np.invert(mask)

    mean_std_dict = {}
    return_dict = {}

    # iterates over channels for masking
    for channel in range(image.shape[-1]):
        masked_image = np.ma.array(image[...,channel], mask = mask)
        channel_mean = round(np.mean(masked_image),2)
        channel_std = round(np.std(masked_image), 2)

        mean_std_dict[f'mean_channel_{channel}'] = channel_mean
        mean_std_dict[f'std_channel_{channel}'] = channel_std

    for key, value in mean_std_dict.items():
        return_dict[key] = float(value)
    return return_dict

def calculate_colocalization(cell_obj, channels, use_mask=False):
    '''calculatew Pearson product-moment correlation coefficients for two channels '''
    image = cell_obj.images_dict['cut_image']
    #if there is no mask the whole image is masked

    if use_mask:
        mask = cell_obj.images_dict['mask']
        mask = np.invert(mask)
    else:
        mask = np.ones_like(image[...,0]).astype(bool)
        mask = np.invert(mask)

    corr_dict = {}
    return_dict = {}

    image_channel_1 = image[...,channels[0]]
    image_channel_2 = image[...,channels[1]]

    masked_image_channel_1 = np.ma.array(image_channel_1, mask = mask)
    masked_image_channel_2 = np.ma.array(image_channel_2, mask = mask)


    p_correlation = pearsonr(masked_image_channel_1.ravel(), masked_image_channel_2.ravel())

    corr_dict['cor_coefficient'] = p_correlation[0]
    corr_dict['p_value'] = p_correlation[1]

    for key, value in corr_dict.items():
        return_dict[key] = round(float(value),2)

    return return_dict

def calc_center_of_mass(cell_obj):
    '''returns center of mass for a given mask for an original image'''
    bounduing_box_x = cell_obj.box_coord[2]
    bounduing_box_y = cell_obj.box_coord[0]

    center_of_mass_coords =  center_of_mass(cell_obj.images_dict['mask'].astype('int'), labels=None, index=None)
    return_dict = {}
    return_dict['center_of_mass_x'] = int(bounduing_box_x + int(center_of_mass_coords[0]))
    return_dict['center_of_mass_y'] = int(bounduing_box_y + int(center_of_mass_coords[1]))
    return return_dict

def create_df(cell_list):
    # get a list of all measured_values names
    column_names = [key for key in cell_list[0].measured_values.keys()]
    cell_objects_data_dict = {}

    for count, cell_obj in enumerate(cell_list):
        values_list = []
        for value in cell_obj.measured_values.values():
            values_list.append(value)
        cell_objects_data_dict[count] = values_list

    df = pd.DataFrame(data=cell_objects_data_dict).transpose()
    df.columns = column_names

    return df
