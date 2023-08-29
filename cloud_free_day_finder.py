import math
import random

import matplotlib.pyplot
import numpy

import splitters

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)


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


def cloud_free_day_finder_visual(year_xa, day_start, day_end, threshold_percent):
    """
    Visualization function, useful for debugging or function tuning
    :param year_xa: one year of xarray data
    :param day_start: first day to consider
    :param day_end: last day to consider
    :param threshold_percent: acceptable smoothness value, lower value, fewer results
    :return: None, should not return anything
    """


    # loading smooth days and their numbers
    smooth_days_xa, smooth_days_numbers = __find_smooth_days(year_xa, day_start, day_end, threshold_percent)

    # taking random day from smooth days list
    random_day = smooth_days_xa[random.randint(0, len(smooth_days_xa))]

    # loading minutes and powers
    random_day = random_day.dropna(dim="minute")
    minutes = random_day.minute.values
    powers = random_day.power.values[0][0]

    # selecting random day number which is outside smooth_day_numbers
    random_day_n = 0
    while True:
        random_day_n = random.randint(day_start, day_end)
        if random_day_n not in smooth_days_numbers:
            break

    # loading year number from xa
    year_n = year_xa.year.values[0]

    # loading random not smooth day from xa
    messy_day = splitters.slice_xa(year_xa, year_n, year_n, random_day_n, random_day_n)
    messy_day = messy_day.dropna(dim="minute")

    # loading plottables from messy day
    messy_day_minutes = messy_day.minute.values
    messy_day_powers = messy_day.power.values[0][0]

    # creating 2 part plot
    fig, axs = matplotlib.pyplot.subplots(2,2)

    # handling plot 1
    axs[0,0].scatter(minutes, powers, s=0.2)
    axs[0,0].set_xlabel("Minute")
    axs[0,0].set_ylabel("Power")
    axs[0,0].set_title("Clear day")

    axs[1,0].scatter(messy_day_minutes, messy_day_powers, s=0.2)
    axs[1,0].set_xlabel("Minute")
    axs[1,0].set_ylabel("Power")
    axs[1, 0].set_title("Cloudy day")

    # right hand size
    # FFT filtering on clear day powers
    clear_day_powers_fft = __fourier_filter(powers, 6)

    messy_day_powers_fft = __fourier_filter(messy_day_powers, 6)



    axs[0, 1].scatter(minutes, clear_day_powers_fft, s=0.2)
    axs[0, 1].set_xlabel("Minute")
    axs[0, 1].set_ylabel("Power")
    axs[0, 1].set_title("Clear day after filtering")

    axs[1, 1].scatter(messy_day_minutes, messy_day_powers_fft, s=0.2)
    axs[1, 1].set_xlabel("Minute")
    axs[1, 1].set_ylabel("Power")
    axs[1, 1].set_title("Cloudy day after filtering")



    matplotlib.pyplot.show()





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
    powers_from_fourier_clean = __fourier_filter(powers, 6)

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


def __fourier_filter(values, values_from_ends):
    """
    :param values: array of values
    :param values_from_ends: how many of the longest frequencies to spare
    :return: values after shorter frequencies are removed
    """

    # FFT based low pass filter
    # Converting values to Fourier transform frequency representatives
    values_fft = numpy.fft.fft(values)
    # values in values_fft represent the frequencies which make up the values array. Structure is as follows:
    # [low, low, ... med, med .... high, high .... med, med .... low,low]
    # this means that by zeroing out most of the values in the center, only the low frequency parts can be chosen


    # zeroing out every value which is further than [values_from_ends] from the ends of the values_fft array
    values_fft[values_from_ends:len(values_fft) - values_from_ends] = [0] * (
            len(values_fft) - 2 * values_from_ends)

    # reversing the fft operation, resulting in values with only low frequency components
    values_ifft = numpy.fft.ifft(values_fft)

    # fft results can be partly imaginary, eq. 2.5 + 2i. Imaginary part should be small as
    values_ifft_real = []

    # saving only real components
    for var in values_ifft:
        values_ifft_real.append(var.real)

    # returning the result of the low pass filter
    return values_ifft_real