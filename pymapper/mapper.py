from functools import partial

import geopandas
import pyproj
import cartopy.crs
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import transform

import pymapper.address_to_coordinate

ax = None
show_grid = False
def create_map():
    # creating global coordinate reference system variable so that it can be used from other functions
    global crs_in_projections
    crs_in_projections = "EPSG:3067"

    # loading background map from shapefile
    background_map = geopandas.read_file("pymapper/shapefiles/maakunnat_siistitty.shp")

    # Create cartopy GeoAxes with proper projection
    etrs89 = cartopy.crs.epsg(3067)  # should be same as crs_in_projection
    global fig
    fig = plt.figure()
    global ax
    __add_wgs84_axis(visible_gridlines=show_grid)

    add_water_details_to_map()

    # plotting base map and adding details
    background_map.plot(ax= ax, color="white", edgecolor="black", alpha=0.2)
    # __add_coarse_water_details(ax)
    # __add_coarse_road_details(ax)


def show_map():
    plt.show()



def plot_point(latitude, longitude, color = "#FF8800", size = 10):
    """
    Adds a scatter point to plot
    :param latitude: in wgs84
    :param longitude: in wgs84
    :return: None
    """
    x, y = __wgs_to_etrs(latitude, longitude)
    plt.scatter(x, y, c= color, s = size)

def plot_address(address_string, color = "#7B68EE", size = 10):

    x, y = pymapper.address_to_coordinate.address_to_coordinate(address_string)
    print(x, y)
    plot_point(x, y, color=color)



def limit_map_to_region(top=61, bottom=59.9, left=23.5, right=26.5):

    global ax
    # boundary points, bottom left,top right corners
    p1 = Point(bottom, left)
    p2 = Point(top, right)

    # projection, this is used to transform the points to correct coordinate system
    projection = partial(
        pyproj.transform,
        pyproj.Proj("EPSG:4326"),
        pyproj.Proj("EPSG:3067")
    )

    # points in correct coord system
    p1a = transform(projection, p1)
    p2a = transform(projection, p2)
    top = p2a.y
    bottom = p1a.y
    left = p1a.x
    right = p2a.x

    """
    print("top")
    print(top)
    print("bottom")
    print(bottom)
    print("left")
    print(left)
    print("right")
    print(right)
    """

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)



def __add_wgs84_axis(visible_gridlines = False):
    global ax
    etrs89 = cartopy.crs.epsg(3067)  # should be same as crs_in_projection
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=etrs89)
    ax.gridlines(draw_labels=True, visible=visible_gridlines)


def __wgs_to_etrs(latitude, longitude):
    p1 = Point(latitude, longitude)
    projection = partial(
        pyproj.transform,
        pyproj.Proj("EPSG:4326"),
        pyproj.Proj("EPSG:3067")
    )
    p1a = transform(projection, p1)

    return p1a.x, p1a.y


def add_title(title):
    plt.title(title)

def toggle_grid(on):
    global show_grid
    show_grid = on

def add_text_to_map(string, latitude, longitude, color="black"):
    x, y = __wgs_to_etrs(latitude, longitude)

    plt.text(x, y, s= string)


def add_water_details_to_map():
    global ax
    details1 = geopandas.read_file("pymapper/shapefiles/simplewaters.shp")
    details1 = details1.to_crs(crs_in_projections)
    details1.plot(ax=ax, color="lightskyblue")

def add__road_details():
    global ax
    details2 = geopandas.read_file("pymapper/shapefiles/mainroads.shp")
    details2 = details2.to_crs(crs_in_projections)
    details2.plot(ax=ax, color="grey", alpha=0.2)