# Bio-image Analysis tools 

![Alt text](https://github.com/Dmitry-Khochanskiy/Cell_extractor_analyser/blob/main/demonstration_figure.png?raw=true)
A flexible pipeline for biological data extraction, analysis and visualization

# Prerequisites
Anaconda
ENV can be loaded from pytorchenv.yml

# Usage
Can be run either from extractor.py script or notebook. All parameters are stored in config.json.
1st step- segmentation and cell images extraction into cell objects. R-CNN and UNet can be used for localization and masking  
2d step - per cell object based analysis    
3d step - per cell object based analysis with ML  
4th step - fancy graphics with seaborn  
5th step - plotting labeled cell objects on the original image, creating montage  

# To do
- [x] Add a complete scripting pipeline 
- [x] Add a script form command line
- [x] Support for configs
- [ ] Add more robust segmentation for default
- [ ] Add support for nDimensional images instead of RGB
- [ ] Implement R-CNN localization and UNet segmentation
- [ ] Possible interactive mode and GUI
- [ ] Code polishing, proper docstrings

# Licence
[MIT](https://choosealicense.com/licenses/mit/)

# Acknowledgments
https://muthu.co/draw-bounding-box-around-contours-skimage/
https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e