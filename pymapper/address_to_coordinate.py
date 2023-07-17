import geopandas
import pandas
from geopy.geocoders import Nominatim


def address_to_coordinate(address):
    """
    :param address: string, for exmaple "Tulliportinkatu 1". Add city name or other info to avoid issues with identical
    street names. For example "tulliportinkatu Kuopio"
    :return: latitude, longitude or NoneType if address was not found
    """

    '''
    Did some digging around and it seems like nominatim has importance values listed for each address 
    'importance': 0.20000999999999994
    These values probably signify which street should be returned when multiple streets have the same name
    '''

    # function based on https://www.geeksforgeeks.org/how-to-get-geolocation-in-python/
    loc = Nominatim(user_agent="GetLoc")

    # entering the location name
    getLoc = loc.geocode(address)

    print(getLoc)
    return getLoc.latitude, getLoc.longitude



def create_geodf_from_addresses(addresses):
    """
    :param addresses: List[] of addresses in text form, street name is enough, street + city if possible
    :return: geopandas geodataframe with addresses is WGS84. Can be plotted
    """

    df = __create_location_dataframe()

    # these are used for keeping track of unfound addresses
    skip_counter = 0
    missing_addresses = []

    for item in addresses:
        geoloc = __get_nominatim_geoloc(item)
        # print(geoloc.raw) returns every field, does not seem to contain a city field unfortunately

        if geoloc is None:
            skip_counter += 1
            missing_addresses.append(item)
            continue

        latitude = geoloc.latitude
        longitude = geoloc.longitude

        # using unknown as city name since geoloc does not provide a city. Could be parsed from the whole address string
        # with some effort though
        df.loc[len(df.index)] = ['Unknown', "Finland", latitude, longitude]

    # dataframe is then transformed into geopandas dataframe
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"  # this sets the datapoints
        # into the "wrong" coordiante system and projection. This is required as coordinates are given as WGS but
        # they have to be later projected into Finnish cylindrical system
    )

    # if addresses could not be found, print debug:
    if skip_counter > 1:
        print("Locations for " + str(skip_counter) + " addresses could not be found. The addresses missing were:")
        print(missing_addresses)
    elif skip_counter == 1:
        print("Location for address \"" + str(missing_addresses[0]) + "\" could not be found.")

    print("Returning " + str(len(addresses) - skip_counter) + " geolocations in geopandas dataframe.")

    # return plottable geopandas dataframe
    return gdf

def __get_nominatim_geoloc(address):
    """
    :param address: text address of a location
    :return:
    """
    loc = Nominatim(user_agent="GetLoc")
    getLoc = loc.geocode(address)
    return getLoc

def __create_location_dataframe():
    # dataframe for datapoints, structure can be modified as long as latitude and longitude still exist
    df = pandas.DataFrame(
        {
            "City": [],
            "Country": [],
            "Latitude": [],
            "Longitude": [],
        }
    )

    return df