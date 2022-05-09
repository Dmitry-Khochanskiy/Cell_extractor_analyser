import pickle
import numpy as np

class CellObj:
    ''' An object which contains object coordinates,bounded image and mask arrays, center coordinates etc'''
    def __init__(self,index, box_coord, image, mask, contour, image_hash):
        self.original_image_hash = image_hash
        self.box_coord = box_coord
        # Saving cut image and mask to a dict
        self.images_dict = {}
        self.images_dict['cut_image'] = image[box_coord[0]:box_coord[1], box_coord[2]:box_coord[3], :]
        self.images_dict['mask'] = mask[box_coord[0]:box_coord[1], box_coord[2]:box_coord[3]]
        self.contour = contour
        self.measured_values = {}

    def return_center_of_box(self):
        return (np.mean(np.arange(self.box_coord[2], self.box_coord[3])),
                np.mean(np.arange(self.box_coord[0], self.box_coord[1]))
                )

    def return_area(self):
        return (np.sum(self.images_dict['mask']))

def add_measured_value(cell_obj_list, function, **kwargs):
    '''Using a separate function adds a value to every cell object in a list'''
    new_cell_obj_list = []

    for cell_obj in cell_obj_list:
        props = function(cell_obj, **kwargs)
        # accepts return as a dictionary, iterates over dictionary and adds it to value dictionary
        for key, value in props.items():
            cell_obj.measured_values[key] = value
        new_cell_obj_list.append(cell_obj)

    return new_cell_obj_list

# Saving CellObj list for on image labeling
def save_objects_as_pickle(cell_list, filepath):
    with open(f'{filepath}/cell_obj_list.pickle', 'wb') as handle:

        pickle.dump(cell_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
