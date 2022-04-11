# Bio-image Analysis tools 

![Alt text](https://github.com/Dmitry-Khochanskiy/Cell_extractor_analyser/blob/main/demonstration_figure.png?raw=true)
A flexible pipeline for biological data extraction, analysis and visualization

# Prerequisites
Anaconda
Env can be loaded from pytorchenv.yml

# Usage
1st step- segmenation and cell images extraction into cell objects. R-CNN and UNet can be used for localization and masking  
2d step - per cell object based analysis    
3d step - per cell object based analysis with ML  
4th step - fancy graphics  
5th step - plotting labeled cell objects on the original image, creating montage  

# To do
- [x] Add a complete scripting pipeline 
- [ ] Add more robust segmentation for default
- [ ] Add support for nDimensional images instead of RGB
- [ ] Implement R-CNN localization and UNet segmentation
- [ ] Possible interactive mode
- [ ] Code polishing, proper docstrings. __main__ in modules 

# Licence
[MIT](https://choosealicense.com/licenses/mit/)

# Acknowledgments
https://muthu.co/draw-bounding-box-around-contours-skimage/
https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e