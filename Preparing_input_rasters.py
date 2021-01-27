
###################################
######## Code to prepare the input rasters
###################################


# Input files
CHborder_path = "D:\\Geodata\\Raw_data\\SwissBOUNDARIES3D\\swissBOUNDARIES3D\\BOUNDARIES_2020\\DATEN\\swissBOUNDARIES3D\\SHAPEFILE_LV95_LN02\\swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.shp"
Sentinel_path = "D:\\Geodata\\Raw_data\\Sentinel_mosaic_CH\\MosaicSentinelCH2018\\SentinelCH2018.tif"
DEM_path = "D:\\Geodata\\Raw_data\\SwissALTI3D\\SwissALTI3D_CH_LV95.tif"
PopDens_path = "D:\\Geodata\\Raw_data\\Population_density\\gd-b-00.03-vz2018statpopb\\STATPOP2018B_B18BTOT_NAto0_noNOLOC.tif"
Building_path = "D:\\Geodata\\Raw_data\\swissTLM3D_2019_gebaeude_footprint\\swissTLM3D_2019_bldng_ftprnt_diss_LV95.shp"
Forest_path = "D:\\Geodata\\Raw_data\\Waldmischungsgrad\\Waldmischungsgrad\\MG2020_0306_oneRF_S1S2_AS_DEM_LV95.tif"
#NDVI_path = "D:\\Geodata\\Raw_data\\Sentinel_NDVI\\S2_2018_NDVI_med_06_08_LV95.tif"
Output_path = "D:\\Current_work\\Projects\\Landscape_typologies\\SwissWide_data\\"


# Import functions and specify settings in ArcGIS
import arcpy, os, rasterio
from arcpy import env
from arcpy.sa import *
import numpy as np
arcpy.SetProduct('ArcInfo')
arcpy.CheckOutExtension('Spatial')
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = 3

arcpy.env.workspace = Output_path

##################################################
# Assign the people per m2 in a building to the buildings in the Building_path shapefile.
##################################################
# Create a Fishnet of the PopDens raster
x_left = arcpy.GetRasterProperties_management(PopDens_path,"LEFT").getOutput(0)
y_bottom = arcpy.GetRasterProperties_management(PopDens_path,"BOTTOM").getOutput(0)
origin = x_left+' '+y_bottom
yAxCoord = x_left+' '+str(int(y_bottom)+10)
cs = arcpy.GetRasterProperties_management(PopDens_path,"CELLSIZEX").getOutput(0)
ncol = arcpy.GetRasterProperties_management(PopDens_path,"COLUMNCOUNT").getOutput(0)
nrow = arcpy.GetRasterProperties_management(PopDens_path,"ROWCOUNT").getOutput(0)
arcpy.CreateFishnet_management(Output_path+"PopDens_fishnet.shp",origin,yAxCoord,cs,cs,nrow,ncol, "#","NO_LABELS","#","POLYGON")

# Intersect the Building_path shapefile with the fishnet.
arcpy.Intersect_analysis(["PopDens_fishnet.shp", Building_path],"Bldng_fishnet_intSect.shp")

# Calculate the area of a building within each fishnet.
arcpy.CalculateGeometryAttributes_management("Bldng_fishnet_intSect.shp",[["Area","AREA"]],"#","SQUARE_METERS")

# Remove parts of buildings that are smaller or equal to 15 m2.
arcpy.MakeFeatureLayer_management("Bldng_fishnet_intSect.shp","bld_fish_lyr")
arcpy.SelectLayerByAttribute_management("bld_fish_lyr",'NEW_SELECTION','"Area" > 15')
arcpy.CopyFeatures_management("bld_fish_lyr", "Bldng_fishnet_intSect_large.shp")

# Calculate the centroid of each building section.
arcpy.FeatureToPoint_management("Bldng_fishnet_intSect_large.shp","Bldng_fishnet_intSect_cntrd.shp","INSIDE")

# Calculate the total building area per PopDens raster cell
arcpy.env.snapRaster = PopDens_path
arcpy.env.extent = PopDens_path
arcpy.env.outputCoordinateSystem = PopDens_path
arcpy.PointToRaster_conversion("Bldng_fishnet_intSect_cntrd.shp","Area","Bldng_area.tif","SUM","#",100)

# Calculate the people per 100 m2 of building in PopDens area.
PopBldngDens = Float(Raster(PopDens_path)) / (Float(Raster("Bldng_area.tif")) / 100)
PopBldngDensNull = IsNull(PopBldngDens)
PopBldngDensCH = Con(PopBldngDensNull, 0, PopBldngDens, "Value = 1")
PopBldngDensCH.save("Pop_Bldng_Dens.tif")
del PopBldngDens, PopBldngDensNull, PopBldngDensCH

# Join the People / 100m2 building with the building footprints
ExtractValuesToPoints("Bldng_fishnet_intSect_cntrd.shp","Pop_Bldng_Dens.tif","Bldng_fishnet_intSect_cntrd_Pp100m2.shp")
arcpy.JoinField_management("Bldng_fishnet_intSect_large.shp", "FID", "Bldng_fishnet_intSect_cntrd_Pp100m2.shp", "ORIG_FID","RASTERVALU")

##################################################
# Make all rasters and shapefiles overlapping.
##################################################

arcpy.env.workspace = Output_path

# Remove Liechtenstein from the boundaries of Switzerland shapefile
arcpy.MakeFeatureLayer_management(CHborder_path,"CHborder_lyr")
arcpy.SelectLayerByAttribute_management("CHborder_lyr",'NEW_SELECTION',"NAME <> 'Liechtenstein'")
arcpy.CopyFeatures_management("CHborder_lyr", "CHborder.shp")

# Clip the Sentinel-RGB image with the borders of Switzerland to get the blue-print raster
sntnl_blueprint = ExtractByMask(Sentinel_path, "CHborder.shp")
sntnl_blueprint.save("SentinelRGB.tif")

# Set the extent, snap and coordinates to the Sentinel blueprint raster
# sntnl_blueprint = Raster("SentinelRGB.tif")
arcpy.env.snapRaster = sntnl_blueprint
arcpy.env.extent = sntnl_blueprint
arcpy.env.outputCoordinateSystem = sntnl_blueprint
arcpy.env.mask = sntnl_blueprint

# Aggregate the DEM raster
DEM = Aggregate(DEM_path, 5, "MEAN")
DEM = Int(DEM)
DEM.save("swissALTI.tif")

# Rasterise the building data
arcpy.PolygonToRaster_conversion("Bldng_fishnet_intSect_large.shp","RASTERVALU","PopDens.tif","CELL_CENTER", "#", 10)
PopDens = Raster("PopDens.tif")
PopDens = Ln(Int(PopDens*100))*100
PopDensNull = IsNull(PopDens)
PopDensCH = Con(PopDensNull, 0, PopDens, "Value = 1")
PopDensCH = ExtractByMask(PopDensCH, sntnl_blueprint)
PopDensCH = Int(PopDensCH)
PopDensCH.save("PopDens_100xLogPpHect.tif")
del PopDens, PopDensNull, PopDensCH, sntnl_blueprint

# Resample the Forest data
cell_size = int(arcpy.GetRasterProperties_management(Sentinel_path,"CELLSIZEX").getOutput(0))
arcpy.Resample_management(Forest_path, "Forest_mixture.tif", cell_size, "NEAREST")
#0 = 100% Laubwald, 10000 = 100% Nadelwald
ForMix = Raster("Forest_mixture.tif")
Deciduous = 10000-ForMix
Coniferous = ForMix
DecidNull = IsNull(Deciduous)
Deciduous = Con(DecidNull, 0, Deciduous, "Value = 1")
Coniferous = Con(DecidNull, 0, Coniferous, "Value = 1")
Deciduous = ExtractByMask(Deciduous, sntnl_blueprint)
Coniferous = ExtractByMask(Coniferous, sntnl_blueprint)
Deciduous = Int(Deciduous)
Coniferous = Int(Coniferous)
Deciduous.save("DeciduousPerc.tif")
Coniferous.save("ConiferousPerc.tif")
del Coniferous, Deciduous, ForMix, DecidNull

##################################################
# Make indices of SentinelRGB
##################################################

# See document "Sentinel_RGB_ratios.docx" for a justification of these ratios
Red = Raster('SentinelRGB.tif\Band_1')
Green = Raster('SentinelRGB.tif\Band_2')
Blue = Raster('SentinelRGB.tif\Band_3')

SentinelRGratio = Int(((Red-Green)/(Red+Green)+1)*1000)
SentinelRBratio = Int(((Red-Blue)/(Red+Blue)+1)*1000)

SentinelRGratio.save("SentinelRGratio.tif")
SentinelRBratio.save("SentinelRBratio.tif")

del SentinelRGratio, SentinelRBratio

# As there are some outliers in this data (i.e. few extreme data points), I reset the top and bottom 0.01 percentiles.
rast_obj = rasterio.open(Output_path+"SentinelRGratio.tif")
out_meta = rast_obj.meta.copy()  
nodata_val = int(rast_obj.nodata)

rast_arr = rast_obj.read(1).astype(float)
rast_arr[rast_arr==nodata_val]=numpy.nan
percentile_001 = np.nanpercentile(rast_arr, 0.01)
percentile_9999 = np.nanpercentile(rast_arr, 99.99)
output = np.where((rast_arr < percentile_001), percentile_001, rast_arr)
output = np.where((rast_arr > percentile_9999), percentile_9999, output)
output = output - percentile_001
output_int = output.astype(int)

with rasterio.open(Output_path+"SentinelRGratio_rescale.tif", "w", **out_meta) as img:
    img.write(output_int, 1)

rast_obj.close()    
    
rast_obj = rasterio.open(Output_path+"SentinelRBratio.tif")
out_meta = rast_obj.meta.copy()  
nodata_val = int(rast_obj.nodata)

rast_arr = rast_obj.read(1).astype(float)
rast_arr[rast_arr==nodata_val]=numpy.nan
percentile_001 = np.nanpercentile(rast_arr, 0.01)
percentile_9999 = np.nanpercentile(rast_arr, 99.99)
output = np.where((rast_arr < percentile_001), percentile_001, rast_arr)
output = np.where((rast_arr > percentile_9999), percentile_9999, output)
output = output - percentile_001
output_int = output.astype(int)

with rasterio.open(Output_path+"SentinelRBratio_rescale.tif", "w", **out_meta) as img:
    img.write(output_int, 1)

rast_obj.close()    
