#-------------------------------------------------------------------------------
# Name:        Create image tiles of a range of raster maps (aerial images and other data)
# Purpose:
#
# Author:      Maarten van Strien
#
# Created:     
# Copyright:   
#-------------------------------------------------------------------------------
##############################
# # Settings
##############################
output_folder = r"D:\Current_work\Projects\Landscape_typologies\SwissWide_data" #The output folder of the tiles
output_filename = "data_tiles_20200706"
input_folder = r"D:\Current_work\Projects\Landscape_typologies\SwissWide_data" #The folder containing the input rasters
#These raster below should have the same origin, resolution and extent!
#The array behind each raster lists the bands that should go into the tiles
#It is assumed that the rasters do not have negative values
#The rasters are presumed to be 16 bit unsigned integers, 
# so if the value range is between 0 and 1 it should be multiplied by e.g. 100
# The value 65535 is reserved for the no-data value.
input_rasters = [["SentinelRGratio_rescale.tif",[0]],
                 ["SentinelRBratio_rescale.tif",[0]],
                 ["swissALTI.tif",[0]],
                 ["PopDens_100xLogPpHect.tif",[0]],
                 ["DeciduousPerc.tif",[0]],
                 ["ConiferousPerc.tif",[0]]] #List with input rasters. 
winSize = 64           #The size of the tiles that will be extracted from the tiles.
stride = 64            #The number of pixels that the moving window should move after every image extraction

##############################
# # Load functions
##############################

import rasterio
import arcpy, os
from arcpy import env
from arcpy.sa import *
import numpy as np
import pandas as pd

##############################
# # Code body
##############################

# Set the python working directory
os.chdir(output_folder)

# Setting the initial environment
env.workspace = output_folder
arcpy.SetProduct('ArcInfo')
arcpy.CheckOutExtension('Spatial')
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = 3

# Extract some input settings
image_stack = []  #Stack of all the images. Each row represents an image.
n_bands = np.sum([len(x[1]) for x in input_rasters]) #The number of raster layers that are included in the final stack.
columns = int(arcpy.GetRasterProperties_management(os.path.join(input_folder, input_rasters[0][0]), "COLUMNCOUNT").getOutput(0)) #Number of columns in input rasters
rows = int(arcpy.GetRasterProperties_management(os.path.join(input_folder, input_rasters[0][0]), "ROWCOUNT").getOutput(0)) #Number of rows in input rasters
np_all_bands = np.zeros((rows,columns,n_bands), 'uint16') #Make an array in which all the raster layers/bands are finally stored.
band_count = 0

# Loop through all the raster layers and aggregate all the bands/layers in one array
for rast in input_rasters:
    rast_tif = os.path.join(input_folder, rast[0])
    rast_obj = rasterio.open(rast_tif)
    nodata_val = int(rast_obj.nodata)
    bands = rast[1]
    for band in bands:
        # Extract the values from the image TIFs. The values are turned to integers.
        np_rast = rast_obj.read(band+1)
        np_rast[np_rast == nodata_val] = 65535
        np_rast = np.uint16(np_rast)
        np_all_bands[:,:,band_count] = np_rast
        band_count += 1
        
del np_rast, rast_tif, rast_obj

# Extract image tiles containing the information in all raster layers/bands
# Image tiles with NA values are removed.
steps_col = np.arange(0,columns-winSize+1,stride) #Determine number of steps in column direction
steps_row = np.arange(0,rows-winSize+1,stride) #Determine number of steps in row direction 
n_tiles = len(steps_col)*len(steps_row)  #Total number of tiles
np_image_stack = np.zeros((n_tiles,winSize,winSize,n_bands), 'uint16') #Numpy array to store all the tiles.  
tile_counter = 0 #Counter for the number of tiles  
# Array to indicate which i-j combination went into the image stack (i.e. without the NA data)
ij_combs = []
 
for i in steps_row:
    for j in steps_col:
        tile_array = np_all_bands[i:i+winSize,j:j+winSize,:]
        if np.any(tile_array==65535):
            continue
        else: 
            np_image_stack[tile_counter,...] = tile_array
            tile_counter += 1
            ij_combs.append([i,j])
        
ij_included = pd.DataFrame(ij_combs, columns=['i','j'])  
np_image_stack = np_image_stack[0:tile_counter]


### Test whether output images can be generated
# from PIL import Image
# im = Image.fromarray(np.uint8(np_image_stack[5999][:,:,0:3]), 'RGB')
# im.save('D:/outfile.jpg')

# Save the image stack to a npz file.
np.savez(output_folder+"/"+output_filename, data=np_image_stack)
# Save ij_included to a csv file.
ij_included.to_csv(output_folder+"/"+output_filename+"_ij_included.csv", index = False)
