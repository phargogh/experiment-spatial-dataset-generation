import math
import random

import numpy
import pygeoprocessing
import pygeoprocessing.kernels
import shapely
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

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

    #import pdb; pdb.set_trace()
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
