import math
import random

import matplotlib.pyplot as plt
import numpy
import pygeoprocessing
from osgeo import osr

AVG_ELEVATION = 1000

dem_elevation = numpy.full((1024, 1024), AVG_ELEVATION,
                           dtype=numpy.float32)

dem_elevation = numpy.full((1, 1), AVG_ELEVATION, dtype=numpy.float32)

resolution = 2  # stating resolution
scale_factor = 2  # Multiplier for how much the resolution grows per iteration
while resolution <= 1024:  # final resolution
    new_dem_elevation = numpy.zeros((resolution, resolution),
                                    dtype=numpy.float32)
    scaled_random_multiplier = 1/math.sqrt(scale_factor)
    for i in range(dem_elevation.shape[1]):
        for j in range(dem_elevation.shape[0]):
            starter_value = dem_elevation[i][j]
            for ii in range(scale_factor):
                for jj in range(scale_factor):
                    new_dem_elevation[i*2+ii][j*2+jj] = starter_value * (
                        1 + (random.random() * scaled_random_multiplier))
    resolution *= scale_factor
    dem_elevation = new_dem_elevation

x = numpy.linspace(0, 1024, 1024)
y = numpy.linspace(0, 1024, 1024)
X, Y = numpy.meshgrid(x, y)
Z = dem_elevation

# Create the 3D plot
fig = plt.figure(figsize=(15, 6))

# Method 1: Surface plot
ax1 = fig.add_subplot(121, projection='3d')
print(X.shape, y.shape, Z.shape)
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Height')
ax1.set_title('3D Surface Plot')
fig.colorbar(surf, ax=ax1, shrink=0.5)

plt.show()

projection_srs = osr.SpatialReference()
projection_srs.ImportFromEPSG(4326)
projection_wkt = projection_srs.ExportToWkt()
pygeoprocessing.numpy_array_to_raster(dem_elevation, -1, (2, -2), (0, 0),
                                      projection_wkt, 'dem.tif')
