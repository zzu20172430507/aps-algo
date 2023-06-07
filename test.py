# -*- coding: cp936 -*-
import arcpy
grd_name = "naqpd02.20000103.grd"
arcpy.RasterToOtherFormat_conversion(grd_name, "E:\\", "TIFF")