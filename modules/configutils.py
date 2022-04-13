import json
#### default config
config = {
    'image_path' :'DAPI_cells.jpg',
    'channel_for_masking': 2,
    'cell_object_list_path' : None,
    'coloc_channels' : [0,2],
    'morph_props' : ('feret_diameter_max', 'perimeter', 'eccentricity'),
    'features_for_ml' : ['feret_diameter_max', 'perimeter', 'eccentricity'],
    'columns_for_plotting' : ['feret_diameter_max', 'perimeter', 'mean_intensity', 'cor_coefficient'],
    'show_step' : 4,
    'values_to_show_on_fig' : ['mean_intensity', 'cor_coefficient', 'KMeans_label']
}

def save_config(config, config_path='config.json'):
    myJSON = json.dumps(config)
    with open("config.json", "w") as jsonfile:
        jsonfile.write(myJSON)
#save_config(config, config_path='config.json')


def check_config(config):
    '''Checks iтput data for validity, to do'''
    return config


def load_config(config_path='config.json'):
    # Loading config from json file
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config
