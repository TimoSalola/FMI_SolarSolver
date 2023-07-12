import math

import numpy

import splitters


def find_smooth_days_xa(year_xa, day_start, day_end, threshold_percent):
    """
    :param year_xa: xarray containing max one year of data
    :param day_start: first day to consider
    :param day_end: last day to consider
    :param threshold_percent: describes the normalized error accepted between a polynomial and real measurements. Use 1
    :return: list of xarray days which satisfy the requirements
    """

    #print(year_xa)
    results = __find_smooth_days(year_xa, day_start, day_end, threshold_percent)
    return results[0]


def find_smooth_days_numbers(year_xa, day_start, day_end, threshold_percent):
    """
    :param year_xa: xarray containing max one year of data
    :param day_start: first day to consider
    :param day_end: last day to consider
    :param threshold_percent: describes the normalized error accepted between a polynomial and real measurements. Use 1
    :return: list of xarray days which satisfy the requirements
    """
    #print(year_xa)
    results = __find_smooth_days(year_xa, day_start, day_end, threshold_percent)
    return results[1]


def __find_smooth_days(year_xa, day_start, day_end, threshold_percent):
    """
    INTERNAL METHOD
    :param year_xa: xarray of one year
    :param day_start: first day to consider
    :param day_end: last day to consider
    :param threshold_percent: smoothness percent, very best days for helsinki dataset are less than 0.4%, 1 gives a good amount of results
    :return: list of xa days and a list of day numbers
    """

    # reading year from year_xa
    # if year_xa contains multiple years worth of data, the first will be chosen. Will most likely break things
    year = year_xa.year.values[0]

    smooth_days_xa = []
    smooth_days_numbers = []

    """
    The loop below goes through every day in given range from year of data
    If the range contains "bad days", this could cause issues. For example a day with zero power for every minute
    This perfectly smooth, but at the same time it's the opposite of what we want
    """
    for day_number in range(day_start, day_end):
        day_xa = splitters.slice_xa(year_xa, year, year, day_number, day_number)

        smoothness_value = __day_smoothness_value(day_xa)

        # print("day:" + str(day_number) + " smoothness: " + str(smoothness_value))
        if smoothness_value < threshold_percent:
            smooth_days_xa.append(day_xa)
            smooth_days_numbers.append(day_number)
        # print("day: " + str(day_number) + " percents off from smooth approximation: " + str(smoothness_value))

    return smooth_days_xa, smooth_days_numbers


############################
#   HELPERS BELOW, ONLY CALL FROM WITHIN THIS FILE
############################


def __day_smoothness_value(day_xa):
    """
    INTERNAL METHOD
    :param day_xa: one day of real measurement data in xa format, has to have fields "minute" and "power"
    :return:  percent value which tells how much longer the distance from point to point is compared to sine/cosine
    fitted curve. Values lower than 1 can be considered good. Returns infinity if too few values in day
    """

    # no values at all, returning infinity
    if len(day_xa["power"].values[0]) == 0:
        return math.inf

    # print("Calculating smoothness value")
    # print(day_xa)
    day_xa = day_xa.dropna(dim="minute")

    # day = day_xa.day.values[0]

    # extracting x and y values
    minutes = day_xa["minute"].values
    powers = day_xa["power"].values[0][0]

    # too few values, returning ab
    if len(powers) < 10:
        return math.inf

    # calculating piecewise distance of measured values
    distances = 0
    for i in range(1, len(minutes)):
        last_x = minutes[i - 1]
        last_y = powers[i - 1]
        this_x = minutes[i]
        this_y = powers[i]

        # for some reason, certain minutes are read as math.inf
        # inf-inf is not well-defined, this needs to be avoided
        if last_y == math.inf or this_y == math.inf:
            continue

        x_delta = last_x-this_x
        x_power = x_delta**2
        y_delta = last_y - this_y
        y_power = y_delta ** 2

        #print("deltay : " +str(y_delta) + " = " + str(last_y) + " - " + str(this_y))
        #print("deltax : " + str(x_delta) + " = " + str(last_x) + " - " + str(this_x))

        distance = math.sqrt(x_power + y_power)
        distances += distance

    # transforming powers into fourier series, removing most values and returning back into time domain

    powers_as_fourier = numpy.fft.fft(powers)
    values_from_ends = 6
    powers_as_fourier[values_from_ends:len(powers_as_fourier) - values_from_ends] = [0] * (
            len(powers_as_fourier) - 2 * values_from_ends)
    powers_from_fourier = numpy.fft.ifft(powers_as_fourier)
    powers_from_fourier_clean = []

    for var in powers_from_fourier:
        powers_from_fourier_clean.append(var.real)

    # this normalizes error in respect to value count
    errors = abs(powers_from_fourier_clean - powers)
    errors_sum = sum(errors)
    errors_normalized = errors_sum / len(powers)

    # if max of powers is 0.0, then division by 0.0 raises errors. If we check max for 0.0 and return infinity
    # our other algorithm should disregard this day completely
    if max(powers) == 0.0:
        return math.inf
    # normalizing in respect to max value and turning into percents
    errors_normalized = (errors_normalized / max(powers)) * 100
    # this line causes occasional errors, some powers lists are just zeros

    return errors_normalized


def __get_measurement_to_poa_delta(xa_day, poa):
    """
    Returns a list of deltas and percentual deltas which can be used for analytics
    :param xa_day: one day of measurements in xarray format
    :param poa: one day of measurements in numpy dataframe
    :return: list of deltas and list of percentual deltas
    """
    # removing the lowest values and nans
    xa_day = xa_day.where(xa_day.power >= 2)
    xa_day = xa_day.dropna(dim="minute")

    # loading poa minutes and powers
    poa_powers = poa["POA"].values
    poa_minutes = poa["minute"].values

    # loading xa minutes and powers
    xa_minutes = xa_day.minute.values
    xa_powers = xa_day.power.values[0][0]

    deltas = []  # absolute value of deltas
    percent_deltas = []  # percentual values of deltas, used for normalizing the errors as 5% at peak is supposed to
    # weight as much as 5% at bottom
    minutes = []  # contains minutes for which deltas were calculated for

    for i in range(len(xa_minutes)):
        xa_minute = xa_minutes[i]  # this is poa index
        xa_power = xa_powers[i]
        poa_power = poa_powers[xa_minute]

        delta = xa_power - poa_power
        percent_delta = (delta / poa_power) * 100
        deltas.append(delta)
        percent_deltas.append(percent_delta)
        minutes.append(xa_minute)

    return deltas, percent_deltas, minutes