def coordinate_to_gmaps_link(latitude, longitude):
    '''
    :param latitude: Latitude in degrees, 64.3242 for example
    :param longitude: Longitude in degrees, 24.231 for example
    :return: Google maps link to that coordinate location
    '''

    # sample link
    # https://www.google.com/maps/@60.6092974,24.8332698,2683m/data=!3m1!1e3?entry=ttu

    zoom_level = "150m"
    # data=!3m1!1e3? marks that the map link is to the satellite image view
    new_link = "https://www.google.com/maps/@" + str(latitude) + "," + str(
        longitude) + "," + zoom_level + "/data=!3m1!1e3"

    return new_link


def print_list_of_links_for_points(points):
    """
    Prints google maps links for each point in points geopandas dataframe by using longitude and latitude values
    Recommended: Sorting points based on latitude or other traits before calling this function
    Site: [1] link: https://www.google.com/maps/@59.9566344,23.3656659,150m/data=!3m1!1e3
    :param points: geopandas dataframe with Longitude and Latitude fields
    :return: None
    """
    for index, point in points.iterrows():
        lat = point.Latitude
        lon = point.Longitude
        print("Site: [" + str(index + 1) + "] link: " + coordinate_to_gmaps_link(lat, lon))