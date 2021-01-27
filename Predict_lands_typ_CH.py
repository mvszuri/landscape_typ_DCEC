# Code to create Switzerland-wide predictions of landscape classes from classes predicted by DCEC.

#Input settings
inFolder = "D:/Current_work/Projects/Landscape_typologies/SwissWide_analysis/Final_analysis/Output10/"
predFile = "DDEC_preds_r5_c45.csv"
outFolder = "D:/Current_work/Projects/Landscape_typologies/SwissWide_analysis/Final_analysis/PostProc/"
blueprint_raster = "D:/Current_work/Projects/Landscape_typologies/SwissWide_data/SentinelRGratio_rescale.tif"
ijInclCSV = "D:/Current_work/Projects/Landscape_typologies/SwissWide_data/data_tiles_20200706_ij_included.csv"
winSize = 64           #The size of the tiles that will be extracted from the tiles.
stride = 64            #The number of pixels that the moving window should move after every image extraction extraction

# Save the feature layer predictions for all the input tiles.
# Import functions and specify settings in ArcGIS
import sys
import rasterio
import arcpy, os
import numpy as np
import pandas as pd
from arcpy import env
from arcpy.sa import *
arcpy.SetProduct('ArcInfo')
arcpy.CheckOutExtension('Spatial')
arcpy.env.overwriteOutput = True
    
# Get the class predictions of the entire dataset.
yPred = np.loadtxt(os.path.join(inFolder,predFile), delimiter=",")

# Extract properties of the blueprint raster necessary for the reconstruction
bp_rast = Raster(blueprint_raster) #Load the raster
columns = int(arcpy.GetRasterProperties_management(bp_rast, "COLUMNCOUNT").getOutput(0)) #Number of columns in input rasters
rows = int(arcpy.GetRasterProperties_management(bp_rast, "ROWCOUNT").getOutput(0)) #Number of rows in input rasters
# lowerLeft = arcpy.Point(bp_rast.extent.XMin, bp_rast.extent.YMin) #Extract coordinates of the lower left corner of the raster
Xmin = bp_rast.extent.XMin
Ymax = bp_rast.extent.YMax

if winSize == stride:
    cell_size_x = int(arcpy.GetRasterProperties_management(bp_rast, "CELLSIZEX").getOutput(0))*winSize #Cell size in x-direction
    cell_size_y = int(arcpy.GetRasterProperties_management(bp_rast, "CELLSIZEY").getOutput(0))*winSize #Cell size in y-direction
else:
    sys.exit("ERROR: winSize is not equal to stride. Use another way to construct the prediction raster")
coord_sys = arcpy.Describe(bp_rast).spatialReference #Coordinate system of the raster

# Get the row and column numbers of all the training tiles
ijIncl = pd.read_csv(ijInclCSV, sep=",")

# Check whether the number of columns in ijIncl is the same as the length of yPred
if ijIncl.shape[0] != len(yPred):
    sys.exit("ERROR: Length ijIncl and yPred not equal")

# Reconstruct the features for the whole study area
steps_col = len(np.arange(0,columns-winSize+1,stride)) #Determine number of steps in column direction
steps_row = len(np.arange(0,rows-winSize+1,stride)) #Determine number of steps in row direction 
n_tiles = steps_col*steps_row  #Total number of tiles
np_features = np.empty([steps_row,steps_col]) #Make an array in which all the features are finally stored.
np_features.fill(numpy.nan)

#Determine coordinates of the lower left corner of the raster
Ymin = Ymax-(steps_row*winSize*10)
lowerLeft = arcpy.Point(Xmin, Ymin) 

#Loop over all the tiles and add the predicted landscape type to np_features.
for row in range(ijIncl.shape[0]):
    i = int(ijIncl.i[row]/winSize)
    j = int(ijIncl.j[row]/winSize)
    pred = yPred[row]
    np_features[i,j] = pred

# Save the array of class labels to a raster.
ag_features = arcpy.NumPyArrayToRaster(np_features[:,:], lowerLeft, cell_size_x, cell_size_y)
arcpy.DefineProjection_management(ag_features, coord_sys)
ag_features = Int(ag_features)
ag_features.save(os.path.join(outFolder, "DEC_class_pred_r5_c45.tif")) 
del ag_features
