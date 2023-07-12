from datetime import datetime

import pandas
from pvlib import location
from pvlib import irradiance
import pandas as pd

import config


############################
#   FUNCTIONS FOR CREATING PLANE OF ARRAY IRRADIANCE CURVES
############################

def get_irradiance_with_multiplier(year, lat, lon, day, tilt, facing, multiplier):
    """
    :param year: Year to simulate for, example: 2021
    :param lat: Geographic latitude of the installation
    :param lon: Geographic longitude of the installation
    :param day: Day of year which the model is created for
    :param tilt: Panel tilt, angle from horizon towards zenith
    :param facing: Horizontal component of panel angle, 0 for north, 90 for east, 180 for south
    :param multiplier: Related to panel area and efficiency. Useful for matching a POA curve with measurements
    :return: Pandas dataframe containing timestamps and irradiance values, resolution 1/minute
    """
    # Creating a pandas dataframe with one day of irradiance data
    irradiance_day = get_irradiance(year, lat, lon, day, tilt, facing)

    # multiplying plane of array irradiance value by multiplier
    irradiance_day["POA"] = irradiance_day["POA"] * multiplier

    return irradiance_day


def get_solar_noon(year, day, latitude, longitude):
    """
    Returns POA approximated solar noon minute
    :param year:
    :param day: day in 1 to 365
    :param latitude: -90 to 90
    :param longitude: -180 to 180
    :return: minute in range 0 to 1440
    """
    fmin, lmin = get_first_and_last_nonzero_minute(latitude, longitude, year, day)

    if fmin is None or lmin is None:
        return None

    solar_noon = (fmin+lmin)/2
    if solar_noon > 1439:
        return solar_noon-1440
    else:
        return solar_noon


def get_first_and_last_nonzero_minute(latitude, longitude, year, day):
    """
    BROKEN FUNCTION
    Returns solar noon minute when longitude, latitude, year and day are known.
    :param latitude:
    :param longitude:
    :param year:
    :param day:
    :return:
    """
    config.YEAR = year
    poa = get_irradiance(year, latitude, longitude, day, 15, 180)
    poa = poa.where(poa["POA"] > 0)
    poa = poa.dropna()

    minutes = poa.minute.values

    #print(len(minutes))

    if len(minutes) > 1420:
        #print("midnight sun, returning none, none")
        return None, None

    if len(minutes) == 0:
        #print("Polar night, returning none as first and last non-zero power minutes")
        return None, None

    first_min = minutes[0]
    last_min = minutes[len(minutes) - 1]

    # 3 easy cases,
    if first_min != 0 and last_min != 1439:
        # ----#####----
        return first_min, last_min

    if first_min == 0 and last_min != 1439:
        # ####-----
        return first_min, last_min

    if first_min != 0 and last_min == 1439:
        # -----####
        return first_min, last_min

    if first_min == 0 and last_min == 1439:
        # #####---##### and ###############
        if len(minutes) == 1440:
            # case #################
            print("midnight sun or equivalent phenomena, returning None, None for first and last minutes of the day")
            return None, None
        else:
            # case #######---######
            for i in range(len(minutes) - 1):
                min1 = minutes[i]
                min2 = minutes[i + 1]
                delta = min2 - min1
                if delta > 1:

                    ## data has a gap

                    #print("data gap,")
                    #print(min1)
                    #print(min2)
                    middle = (min1+min2)/2

                    if middle > 1440/2:
                        # gap happens on end part
                        return min2-1440, min1

                    if middle <= 1440/2:
                        # gap in beginning
                        return min2, min1+1440

                    return None, None



def get_irradiance(year, lat, lon, day, tilt, facing):
    """
    Main irradiance estimation function. Based on code from pvlib tutorial:
    https://pvlib-python.readthedocs.io/en/stable/gallery/irradiance-transposition/plot_ghi_transposition.html
    """

    # creating site data required by pvlib poa
    tz = 'GMT'  # assuming that measurements are in UTZ GMT time
    #year = config.YEAR  # loading year from config file
    site = location.Location(lat, lon, tz=tz)

    # creating a pandas entity containing the times for which the irradiance is modeled for
    date = datetime.strptime(str(year) + "-" + str(day), "%Y-%j").strftime("%m-%d-%Y")

    times = pd.date_range(date,  # year + day for which the irradiance is calculated
                          freq='1min',  # take measurement every 1 minute
                          periods=60 * 24,  # how many measurements, 60 * 24 for 60 times per 24 hours = 1440
                          tz=site.tz)  # timezone, using gmt

    # creating a clear sky and solar position entities
    clearsky = site.get_clearsky(times)
    solar_position = site.get_solarposition(times=times)

    # creating PVlib plane of array irradiance dataframe
    POA_irradiance = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=facing,
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'])

    # turning the times dataframe to a list of minutes
    times = clearsky.index.time
    minutes = []
    for time in times:
        minutes.append(time.hour * 60 + time.minute)

    # creating the output dataframe which consists of only necessary data, minutes and corresponding poa values
    output_df = pd.DataFrame(
        {
            "minute": minutes,
            'POA': POA_irradiance['poa_global']
        }
    )

    return output_df


def create_poa_df_for_year(year, lat, lon, tilt, facing):
    """
    :param year:   year to create poa for
    :param lat:     latitude
    :param lon:     longitude
    :param tilt:    panel tilt
    :param facing:  panel facing
    :return: pandas dataframe containing 365 poa models, one for each day of the year with given parameters
    """
    config.YEAR = year
    return create_poa_df_for_range(range(0, 365), lat, lon, tilt, facing)


def create_poa_df_for_range(list_of_day_numbers, lat, lon, tilt, facing):
    """
    Creates a POA DF for each day in list of day numbers, using given latitude, longitude, tilt and facing
    """

    poa_days = []
    for day in list_of_day_numbers:
        poa_day = get_irradiance(lat, lon, day, tilt, facing)
        poa_day["day"] = day
        poa_days.append(poa_day)

    year_poa_df = pandas.concat(poa_days)

    return year_poa_df
