import statistics

import pvlib_poa
import splitters


def longitude_from_solar_noon_solar_noon_poa(long0, solar_noon, solar_noon_poa):
    """
    Improved longitude estimation function
    :param long0: longitude used for simulating solar_noon_poa
    :param solar_noon: measured solar noon minute
    :param solar_noon_poa: simulated solar noon minute
    :return: estimated longitude
    """
    return long0 - (360 / 1440) * (solar_noon - solar_noon_poa)


def estimate_longitude_based_on_year(year_xa):
    """
    Estimates the longitude of a solar PV installation when one year of data is given.
    Hard coded values
    simulation_longitude = 25
    simulation_latitude = 60
    Should be changed to better reflect the region where the installations are expected to be if the algorithm is used
    for installations outside of Finland

    :param year_xa: One year long of xarray data
    :return: estimated longitude
    """

    # reading year from year_xa
    year = year_xa.year.values[0]
    # reading days from year_xa
    days = year_xa.day.values

    # listing simulation parameters, CHANGE THESE IF
    simulation_longitude = 25
    simulation_latitude = 60

    # list for simulated longitude values, needed as one value is simulated for each day
    longitudes = []

    # calculating a longitude for each day in year_xa
    for day in days:
        # spitting needed day from xa
        day_xa = splitters.slice_xa(year_xa,year, year,day, day)
        day_xa = day_xa.dropna(dim="minute")

        # taking first and last minute values
        fmin, lmin = __xa_dirty_get_first_last_minute_of_solar_output(day_xa)
        if fmin is None or lmin is None:
            # skipping if returned none
            continue

        # estimating solar noon based on them
        estimated_solar_noon = (fmin+lmin)/2

        # simulating solar noon minute
        simulated_solar_noon = pvlib_poa.get_solar_noon(year, day, simulation_latitude, simulation_longitude)

        # estimating longitude with the help of estimated solar noon, simulated solar noon and simulated solar noon parameters
        estimated_longitude = longitude_from_solar_noon_solar_noon_poa(simulation_longitude, estimated_solar_noon, simulated_solar_noon)
        longitudes.append(estimated_longitude)

    # returning statistical mean of estimated longitudes
    return statistics.mean(longitudes)


def __xa_dirty_get_first_last_minute_of_solar_output(xa_day):
    """
    TAKES A SINGLE DAY FROM XA AND TRIES TO FIGURE OUT THE FIRST AND LAST VALID MINUTE IN IT.
    FIRST OUTPUT VALUE SHOULD BEGIN A BLOCK AND SECOND SHOULD END IT, THIS RESULTS IN A RATHER LONG FUNCTION
    """
    xa_day = xa_day.dropna(dim="minute")

    minutes = xa_day.minute.values

    # returning None, None pair for inputs which might result in invalid values
    if len(minutes) / 1440 > 0.9:
        '''
        print("At day " + str(xa_day.day.values) + " Too many minutes in list, could be indication of a full day, "
                                                   "sun doesn't set low enough?")'''
        return None, None

    if len(minutes) / 1440 < 0.3:
        '''
        print(
            "At day " + str(xa_day.day.values) + " Too few measurements, day length is too short which could mean too "
                                                 "early spring or too late fall day.",
            "It's also possible that there are gaps in the data. Preprocessing data further or skipping this day is"
            " recommended.")
        '''
        return None, None

    # Finding the longest gaps in data and their start/end points
    longest_gap = 0
    second_longest_gap = 0
    lgap1 = 0
    lgap2 = 0

    for i in range(len(minutes) - 1):
        p1 = minutes[i]
        p2 = minutes[i + 1]

        gap = p2 - p1

        if gap > longest_gap:
            second_longest_gap = longest_gap
            longest_gap = gap
            lgap1 = p1
            lgap2 = p2
        elif gap > second_longest_gap:
            second_longest_gap = gap

    # SWITCH CASE TYPE IF CONSTRUCTION
    # GOAL: return likely start and end point of light period., end might be over 1440 because of timezones

    if longest_gap == second_longest_gap == 1:
        # well-behaved central block, 000000000+++++++++000000000
        # print("most likely a well-behaved central block")
        return minutes[0], minutes[len(minutes) - 1]  # should be able to return these values

    elif longest_gap > 100 and second_longest_gap < 3:
        # well-behaved block, +++++++++000000000000+++++++++++
        # or not so well, 000000+++++00000++++++0000000

        if minutes[0] < 3 and minutes[len(minutes)] > 1438:  # +++++++++000000000000+++++++++++
            print("most likely well behaved block, timezone causes split into 2sections")
            return lgap2, 1440 + lgap1
        elif minutes[0] > 3:  # 000+++++++00000000+++++
            print("Unable to parse")
            return None, None  # unable to parse

    elif longest_gap > 100 and second_longest_gap < 15:
        # reasonably well behaved 2 part block, +++++++++0000000+++++++++++++++
        print("reasonably well behaved 2 part block")
        if minutes[0] < 10 and minutes[len(minutes) - 1] > 1420:
            print("likely to be reasonably well behaving block")
            return lgap2, 1440 + lgap1

    # returning none. none if longer gaps present
    elif 100 > longest_gap > 15 and 100 > second_longest_gap > 15:
        print("badly behaving, at least 2 medium sized gaps")
        return None, None

    # code should never get this far, this is a debug message that should only be printed if something goes wrong
    print("Gaps were " + str(longest_gap) + " and " + str(second_longest_gap))

    print("WARNING ####################################################################################")
    print("###############Didn't activate any of the cases#############################################")
    print("CHECK geoguesser_longitude.py FUNCTION __xa_dirty_get_first_last_minute_of_solar_output(xa_day)")

    return None, None
