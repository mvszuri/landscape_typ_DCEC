# landscape_typ_DCEC
Python code to create automated landscape typologies with Deep Convolutional Embedded Clustering.

The code was used to generate the results presented in the manuscript titled "Unsupervised deep learning of landscape typologies from remote sensing imagery and other spatial data" (version 7 April 2021).

Parts of the code were adapted from the code belonging to the publication: Guo X, Liu X, Zhu E, Yin J (2017) Deep Clustering with Convolutional Autoencoders, Proceedings of the International Conference on Neural Information Processing, Guangzhou, China.
Gou et al.'s code can be found here: https://github.com/XifengGuo/DCEC.

## Details of the individual files:
- Preparing_input_rasters.py: Code to prepare the Switzerland-wide input rasters (satellite images and other data).
- Create_raster_tile_training_set.py: Code to create image tiles of the Switzerland-wide input rasters.
- DCEC_autoencoder.py: Code to perform Deep Convolutional Embeded Clustering on the image tiles.
- ClustLay_DCEC.py & Functions_DCEC.py: Functions required to perform Deep Convolutional Embeded Clustering.
- Predict_lands_typ_CH.py: Code to create Switzerland-wide predictions of landscape classes from classes predicted by DCEC.

