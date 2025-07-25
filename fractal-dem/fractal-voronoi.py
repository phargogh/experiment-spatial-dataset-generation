import logging
import math
import random
import shutil

import matplotlib.pyplot as plt
import numpy
import pygeoprocessing
import pygeoprocessing.kernels
import pygeoprocessing.routing
import shapely
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

logging.basicConfig(level=logging.DEBUG)

AVG_ELEVATION = 1000
SRS = osr.SpatialReference()
SRS.ImportFromEPSG(4326)
WKT = SRS.ExportToWkt()

width = 300
height = 120
dem_polygon = shapely.geometry.box(0, 0, width, height)

points_per_polygon = 7

array = numpy.zeros((height, width), dtype=numpy.float32)

polygon_heights = {}


def adjustment(depth):
    return 1 + (random.random() * (1/math.sqrt(depth)))


def recurse(polygon, elevation, depth):
    elevation = elevation * adjustment(depth)
    if polygon.area <= 1:  # pixelsize
        centroid = polygon.centroid
        array_x = math.floor(centroid.x)
        array_y = math.floor(centroid.y)

        array[array_y][array_x] = elevation
        return

    points_dropped = 0
    points = []
    while True:
        if points_dropped >= points_per_polygon:
            break

        random_x = random.uniform(
            polygon.bounds[0], polygon.bounds[2])
        random_y = random.uniform(
            polygon.bounds[1], polygon.bounds[3])

        new_point = shapely.geometry.Point(random_x, random_y)
        if not new_point.within(polygon):
            # Try again
            continue

        points_dropped += 1
        points.append(new_point)

    # create voronoi geometries
    points = shapely.geometry.MultiPoint(points)
    voronoi_geometry = shapely.geometry.MultiPolygon(
        shapely.voronoi_polygons(points, extend_to=polygon))

    #pygeoprocessing.shapely_geometry_to_vector(
    #    [points, polygon, voronoi_geometry],
    #    f'vector-{depth}.geojson', WKT, 'GeoJSON',
    #    fields={'fid': ogr.OFTInteger},
    #    attribute_list=[{'fid': 1}, {'fid': 2}, {'fid': 3}],
    #    ogr_geom_type=ogr.wkbUnknown)

    for voronoi_geom in voronoi_geometry.geoms:
        recurse(voronoi_geom.intersection(polygon), elevation, depth + 1)


recurse(dem_polygon, AVG_ELEVATION, 1)


# Locate any pixels that weren't updated, take the average of nearby pixels
print(numpy.sum(array == 0))

pygeoprocessing.numpy_array_to_raster(
    array, -1, (2, -2), (0, 0), WKT, 'dem-voronoi.tif')

pygeoprocessing.kernels.linear_decay_kernel('kernel.tif', 5)

pygeoprocessing.convolve_2d(('dem-voronoi.tif', 1), ('kernel.tif', 1),
                            'convolved-voronoi.tif', mask_nodata=False,
                            ignore_nodata_and_edges=True)

shutil.copyfile('convolved-voronoi.tif', 'convolved-voronoi-pow.tif')
with gdal.Open('convolved-voronoi-pow.tif', gdal.GA_Update) as ds:
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()

    # Raise to a power to raise peaks and deepen valleys
    array_pow = array ** 5.3

    # Rescale to a reasonable range, say, 0 - 2,000
    rescaled = (array_pow / (array_pow.max() - array_pow.min())) * 2000

    band.WriteArray(rescaled)

pygeoprocessing.routing.fill_pits(
    ('convolved-voronoi-pow.tif', 1), 'voronoi-d8-filled.tif')
pygeoprocessing.routing.flow_dir_d8(
    ('voronoi-d8-filled.tif', 1), 'voronoi-d8-dir.tif')
pygeoprocessing.routing.flow_accumulation_d8(
    ('voronoi-d8-dir.tif', 1), 'voronoi-d8-accum.tif')
pygeoprocessing.routing.extract_streams_d8(
    ('voronoi-d8-accum.tif', 1), 50, 'voronoi-d8-streams.tif')


x = numpy.linspace(0, width, width)
y = numpy.linspace(0, height, height)
X, Y = numpy.meshgrid(x, y)
Z = rescaled

# Create the 3D plot
fig = plt.figure(figsize=(20, 6))

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
