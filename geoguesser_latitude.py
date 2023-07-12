import numpy

import pvlib_poa
import splitters



###############################################################
#   Latitude estimation function
###############################################################


def slopematch_estimate_latitude_using_single_year(xa, year, first_day, last_day):
    """
    :param xa: xarray file following the structure described in solar power data loader
    :param year: year, eq. 2021
    :param first_day: first day of analysis interval, use 125 if
    :param last_day: last day of analysis interval, use 250
    :return: [first minute based latitude estimation, last minute based latitude est]
    """

    # taking a slice from given measurements xa
    year_data = splitters.slice_xa(xa, year, year, first_day, last_day)

    # creating a set of poa based 3rd degree polynomials, input is slope angle, output is latitude
    # this is slow-ish to compute
    model_first_mins, model_last_mins = __get_poa_slope_models_for_day_ranges(year, first_day, last_day, 55, 75)

    # creating first degree models from measurement data
    measured_model_first_mins, measured_model_last_mins, days = __get_measurements_minute_models_at_days(year_data, year, first_day,
                                                                                                         last_day)

    # estimating latitude from first and last degree models and measurement slopes
    # this happens by giving the slope of measurements from the earlier 1st degree models to the 3rd degree models
    latitude_firsts = __3rd_degree_poly_at_x(model_first_mins, measured_model_first_mins[1])
    latitude_lasts = __3rd_degree_poly_at_x(model_last_mins, measured_model_last_mins[1])

    #print("Single year " + str(year) + " estimated latitudes:")
    #print(latitude_firsts)
    #print(latitude_lasts)

    return latitude_firsts, latitude_lasts


###############################################################
#   Helpers below
###############################################################

def __get_poa_slope_models_for_day_ranges(year, first_day, last_day, latitude_low, latitude_high):
    """
    returns 3rd degree polynomial models, the input of which should be the measured slope,

    :param year: year to generate model for
    :param first_day: first day in day range, 250 recommended
    :param last_day: last day in day range, 300 recommended
    :param latitude_low:
    :param latitude_high:
    :return:
    """

    slopes_firsts = []
    slopes_lasts = []

    latitudes = []

    for latitude in range(latitude_low, latitude_high+1):
        fmins = []
        days = []
        lmins = []
        for d in range(first_day, last_day+1, 10):
            fmin, lmin = pvlib_poa.get_first_and_last_nonzero_minute(latitude, 0, year, d)
            if fmin is None or lmin is None:
                continue
            fmins.append(fmin)
            lmins.append(lmin)
            days.append(d)

        first_minutes_model = numpy.polynomial.polynomial.polyfit(days, fmins, 1)
        last_minutes_model = numpy.polynomial.polynomial.polyfit(days, lmins, 1)
        slopes_firsts.append(first_minutes_model[1])
        slopes_lasts.append(last_minutes_model[1])
        latitudes.append(latitude)

    first_model = numpy.polynomial.polynomial.polyfit(slopes_firsts, latitudes, 3)
    last_model = numpy.polynomial.polynomial.polyfit(slopes_lasts, latitudes, 3)
    return first_model, last_model

def __get_measurements_minute_models_at_days(xa, year, day_start, day_end):
    """
    :param xa: XA containing PV installation power output data in predefined format
    :param year: year to create model for
    :param day_start: first day to use
    :param day_end: last day to use
    :return: fmin_model, lminmodel, a pair of linear models. These are lists, [offset, derivative]
    """
    # taking section
    correct_days = splitters.slice_xa(xa, year, year, day_start, day_end)
    # splitting section into first, last and days
    first_mins, last_mins, days = __xa_slice_to_first_last_and_days(correct_days, year)
    # fitting linear equations
    last_minutes_model = numpy.polynomial.polynomial.polyfit(days, last_mins, 1)
    first_minutes_model = numpy.polynomial.polynomial.polyfit(days, first_mins, 1)

    return first_minutes_model, last_minutes_model, days

def __xa_slice_to_first_last_and_days(xa_slice, year):
    first_minutes = []
    last_minutes = []
    days = []

    for day in xa_slice["day"].values:
        xa_day = splitters.slice_xa(xa_slice, year, year, day, day)
        xa_day = xa_day.dropna(dim="minute")

        minutes = xa_day.minute.values
        # print(minutes)

        first, last = __minute_list_to_first_last(minutes)

        if first is not None:
            first_minutes.append(first)
            last_minutes.append(last)
            days.append(day)

    return first_minutes, last_minutes, days

def __minute_list_to_first_last(minutelist):
    """
    Translates a list of minute numbers to last and first minutes of the day. Error-prone function.
    :param minutelist:
    :return: first, last -minute of the day. Will return None, None if issues encountered
    """

    ##############################
    # FUNCTION STEPS
    # 1. CALCULATE LENGTH OF GIVEN LIST
    # 2. REJECT IF LEN IS BAD
    # 3. RETURN FIRST AND LAST MINUTES IF THEY ARE OBVIOUS
    # 4. EXAMINE DIFFICULT CASES ON CASE BY CASE BASIS
    # 5. RETURN NONE, NONE IF NONE OF THE CASES APPLIED
    ##############################

    # 1. CALCULATE LENGTH OF GIVEN LIST
    minutelist_len = len(minutelist)

    # 2. REJECT IF LEN IS BAD
    if minutelist_len > 1200:
        # "too much data", the longest days of the year might be unreliable
        return None, "Too much data. Minutelist len: " + str(minutelist_len)

    if minutelist_len < 200:
        # "too little data", the shortest days of the year might be unreliable
        return None, "Too little data. Minutelist len: " + str(minutelist_len)

    # 3. RETURN FIRST AND LAST MINUTES IF THEY ARE OBVIOUS
    first_value = minutelist[0]
    last_value = minutelist[minutelist_len - 1]

    minutelist_internal_len = last_value - first_value + 1
    # internal len, [3,4,5,6,7,8] = 8-3+1 = 6 = minutelist_len
    # internal len might not be same as len,

    if minutelist_internal_len == minutelist_len:
        # CASES:  ____############___ no gaps
        #  ###########_____ no gaps
        # ______############ no gaps
        return first_value, last_value

    # 4. EXAMINE DIFFICULT CASES ON CASE BY CASE BASIS
    # NOTE THAT THESE HAVE NOT BEEN TESTED WITH REAL DATASETS, THIS IS THE ERROR PRONE UNTESTED SECTION
    # FMI HELSINKI AND KUOPIO DATASETS DO NOT SEEM TO REACH THIS SECTION AT ALL

    # calculating helper data
    gaps_in_minutes = []
    gap_minutes = []
    # will return 1 if first minute is 1 and last 1440, will return 1 if first is 0 and last is 1439
    # this end gap is important for finding out if end gap exists
    end_gap = first_value + (1440 - last_value)
    if end_gap > 1:
        gap_minutes.append([last_value, first_value])  # note that last is before first because
        #       day1,  | day2
        #  ___#####___|___####___
        #         ^------^ is more in line with the gap type in the gap for loop
        # than^---^
        gaps_in_minutes.append(end_gap)

    for i in range(1, minutelist_len):
        # gap between minute and the previous minute
        gap = minutelist[i] - minutelist[i - 1]
        if gap > 1:
            # here gaps are added from first to last,
            # because ####____####
            #             ^--^
            gap_minutes.append([minutelist[i - 1], minutelist[i]])
            gaps_in_minutes.append(gap)

    print("Gaps in minutes:")
    print(gaps_in_minutes)
    print("gaps end")

    if len(gaps_in_minutes) == 1:
        # case __#####__ was already handled, this 1 gap case must be ####____####
        # returning first minute and last minute
        return gaps_in_minutes[0][1], gap_minutes[0][0] + 1439

    # missing multiple gap processing, will always return none, none on multi gap datasets

    # 5. RETURN NONE, NONE IF NONE OF THE CASES APPLIED
    return None, "No cases applied"

def __3rd_degree_poly_at_x(poly, x):
    """
    :param poly: polynomial model from numpy.polynomial.polynomial.polyfit, 3rd degree or higher
    :param x: intended to be slope angle
    :return: intended to return estimated latitude
    """
    return poly[0] + poly[1] * x + poly[2] * (x ** 2) + poly[3] * (x ** 3)