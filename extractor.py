from modules.extraction import *
from  modules.imgutil import *
from  modules.measurment import *
from  modules.mlmeasurment import *
from modules.plots import *
from  modules.cellvis import *
from modules.configutils import *

def main():

# Load config
    config = load_config(config_path='config.json')
# Image loading, cell extraction, primiry filtering and creating a lost of cell objects
    image, image_hash = image_loader(config['image_path'])
    image_name = config['image_path'].rsplit('.', 1)[0]
    path_to_save = ('results/' + f'{image_name}')
    show_image(image, "original image")
# any localisation and thresholding algorithm can be put here

    mask = thresholder(image,channel=config['channel_for_masking'])
    show_image(mask, "mask")
    contours, bounding_boxes = object_bounding_boxes(mask,detector,1)
    cell_list = create_cell_list(bounding_boxes, image, mask, image_hash, contours)
    cell_list  = filter_cell_objects_by_size(cell_list, image, show_hist=True, quantile=0.05)


# Per cell measurment with custom metrics
#Opening file and unpickling its content if needed
    if config['cell_object_list_path']:
        with open(config['cell_object_list_path'] , 'rb') as pickled_file:
            cell_list = pickle.load(pickled_file)

# Sequantial calculating data for cell objects
    cell_list = add_measured_value(cell_list, calculate_morpho_properties,properties=(config['morph_props']))
    cell_list = add_measured_value(cell_list, calc_center_of_mass)
    cell_list = add_measured_value(cell_list, calculate_intensity)
    cell_list = add_measured_value(cell_list, calculate_colocalization, channels=config['coloc_channels'], use_mask=True)

    df = create_df(cell_list)

    df.to_csv(f'{path_to_save}/measurments.csv', index=False)

# Model traning if needed and per cell ML measurment
# Training a model on whole image data set
    model, scaler =  fit_KMeans(df[config['features_for_ml']])
# per cell model inference
    new_cell_list = add_measured_value(cell_list, calculate_label_KMeans, model=model,
                                   scaler=scaler,  features=config['features_for_ml'])
    save_objects_as_pickle(cell_list, f'{path_to_save}')

# Plotting features
    columns_list = df.columns
    pair_grid(df, config['columns_for_plotting'], path_to_save)
    heat_map(df, config['columns_for_plotting'], path_to_save)
    correlation_heat_map(df, config['columns_for_plotting'], path_to_save)

# Visualization on image and cells montage creation
    labeling_image(image, image_hash, cell_list[::config['show_step']], path_to_save, config['values_to_show_on_fig'])
    make_montage(cell_list,config['values_to_show_on_fig'] ,path_to_save, 3)

if __name__ == "__main__":
    main()
