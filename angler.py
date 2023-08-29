import math

import matplotlib.pyplot
import numpy
import config
import multiplier_matcher
import pvlib_poa
import splitters


############################
#   FUNCTIONS FOR ESTIMATING PANEL ANGLES
#   USE find_best_multiplier_for_poa_to_match_single_day_using_integral FOR COMPUTING THE MULTIPLIER IF ANGLES ARE KNOWN
#   OTHER __FUNCTIONS ARE HELPERS, INTENDED FOR INTERNAL USE
############################


def test_single_pair_of_angles(day_xa, known_latitude, known_longitude, tilt, facing):
    """
    Tests a pair of angles against a day of measurements. Returns a fitness value, lower is better
    :param day_xa: xarrya containing one day of measurements
    :param known_latitude: latitude coordinate of installation in wgs84
    :param known_longitude: longitude coodrinate of installation in wgs84
    :param tilt: test tilt angle
    :param facing: test facing angle
    :return: fitness value, lower is better
    """

    # reading year and day from xa
    day_n = day_xa.day.values[0]
    year_n = day_xa.year.values[0]

    # creating initial poa
    poa_initial = pvlib_poa.get_irradiance(year_n, known_latitude, known_longitude, day_n, tilt, facing)

    # matching poa with single segment integral method
    multiplier = find_best_multiplier_for_poa_to_match_single_day_using_integral(day_xa, poa_initial)

    # matching multiplier
    poa_initial["POA"] = poa_initial["POA"] * multiplier

    # comparing multiplier matched with measurements for fitness value, lower is better
    fitness = get_measurement_to_poa_delta_cost(day_xa, poa_initial)

    return fitness


def take_poa_and_return_multiplied_poa_best_matching_measurements(xa_day, poa):
    """
    Takes measurements and simulation, returns simulation scaled to match measurements
    :param xa_day: xa containing one day of measurements
    :param poa: simulated plane of array irradiance curve
    :return: area matched poa curve
    """
    multiplier = find_best_multiplier_for_poa_to_match_single_day_using_integral(xa_day, poa)
    poa["POA"] = poa["POA"] * multiplier
    return poa


def get_measurement_to_poa_delta_cost(xa_day, poa):
    """
    Returns the delta between measurements and simualted values
    :param xa_day:
    :param poa:
    :return:
    """

    deltas, percents, minutes = __get_measurement_to_poa_delta(xa_day, poa)

    # print(percents)

    #percentual_cost = sum([abs(ele) for ele in percents]) / len(percents)

    #print(poa)
    #print(xa_day)

    abs_deltas = sum([abs(ele) for ele in deltas])

    return  abs_deltas #/len(deltas)



def find_best_multiplier_for_poa_to_match_single_day_using_integral(day_xa, poa):
    # Steps:
    # Find out common minutes
    # Calculate powers for xa and poa for those common minutes
    # calculate the ratio of these powers
    # use ratio as a multiplier
    # This should be similar to taking the integral over common points and comparing the areas

    # dropping na
    xa2 = day_xa.dropna(dim="minute")

    # getting area under measurements and the minutes that were used
    xa_minutes = xa2["minute"].values
    xa_powers = xa2["power"].values[0][0]
    sum_of_measured_powers = sum(xa_powers)

    # assuming that there are no gaps in the data
    first_minute = xa_minutes[0]
    last_minute = xa_minutes[len(xa_minutes) - 1]

    # print(" selecting interval " + str(first_minute) + " to " + str(last_minute))

    poa2 = poa.where(poa.minute >= first_minute)
    poa3 = poa2.where(poa.minute <= last_minute)
    poa4 = poa3.dropna()

    poa_sum_of_poa = sum(poa4["POA"])

    # this can be used as a multiplier to compute the required multiplier for the poa to match the measurements
    ratio = sum_of_measured_powers / poa_sum_of_poa

    return ratio


def get_multiplier_matched_poa(day_xa, latitude, longitude, tilt, azimuth):
    # day and year numbers from Xarray
    day_n = day_xa.day.values[0]
    year_n = day_xa.year.values[0]
    poa = pvlib_poa.get_irradiance(year_n, latitude, longitude, day_n, tilt, azimuth)
    multiplier = multiplier_matcher.get_estimated_multiplier_for_day(day_xa, poa)
    poa = pvlib_poa.get_irradiance_with_multiplier(year_n, latitude, longitude, day_n, tilt, azimuth, multiplier)

    return poa


def test_n_fibonacchi_sample_fitnesses_against_day(day_xa, samples, latitude, longitude):
    """
    Takes one good xa, known geolocation and sample count, returns tested angles and their fitnesses
    :param day_xa: cloud free xa
    :param samples: sample count, higher means more angle pairs will be tested
    :param latitude: known installation latitude coordinate
    :param longitude: known installation longitude coordinate
    :return: return [tilts(rad)], [azimuths(rad)], [fitnessess(decimal, lower better)]
    """

    # day and year numbers from Xarray
    day_n = day_xa.day.values[0]
    year_n = day_xa.year.values[0]

    # tilt and azimuth values on a fibonacci half sphere
    tilts_rad, azimuths_rad = get_fibonacci_distribution_tilts_azimuths(samples)

    delta = angle_distance_between_points(numpy.degrees(tilts_rad[0]), numpy.degrees(azimuths_rad[0]), numpy.degrees(tilts_rad[1]), numpy.degrees(azimuths_rad[1]))
    print("estimating grid density, assuming points are spread evenly there should be a point every " + str(round(delta,4)) + " degrees")

    # tilt and azimuth values in degrees
    tilts_deg, azimuths_deg = [], []
    for i in range(len(tilts_rad)):
        tilts_deg.append((tilts_rad[i] / (2 * math.pi)) * 360)
        azimuths_deg.append((azimuths_rad[i] / (2 * math.pi)) * 360)

    # debugging messages
    #print("testing fitnessess at tilts:")
    #print(tilts_deg)
    #print("and azimuths")
    #print(azimuths_deg)

    # creating poa at tilt+azimuth and fitness
    fitnesses = []

    for i in range(len(tilts_deg)):
        tilt = tilts_deg[i]
        azimuth = azimuths_deg[i]
        fitness = test_single_pair_of_angles(day_xa, latitude, longitude, tilt, azimuth)
        fitnesses.append(fitness)

    return tilts_rad, azimuths_rad, fitnesses


############################
#   GLOBAL HELPERS
############################


def get_best_fitness_out_of_results(tilt_rads, azimuth_rads, fitnesses):
    """
    Returns tilt and azimuth values for lowest fitness value
    :param tilt_rads:
    :param azimuth_rads:
    :param fitnesses:
    :return: best_azimuth, best_tilt, best_fitness
    """

    best_azimuth = 0
    best_tilt = 0
    best_fit = math.inf
    for i in range(len(tilt_rads)):
        azimuth = azimuth_rads[i]
        tilt = tilt_rads[i]
        fitness = fitnesses[i]
        if fitness < best_fit:
            best_azimuth = azimuth
            best_tilt = tilt
            best_fit = fitness

    return best_tilt, best_azimuth, best_fit



def get_fibonacci_distribution_tilts_azimuths_near_coordinate(tilt, azimuth, samples_total, distance):
    """
    Unused function, generates local lattices
    Returns fibonacci lattice points which are closer than (distance) from (tilt) and (azimuth) in cartesian space
    :param tilt: Tilt angle in degrees
    :param azimuth: Azimuth angle in degrees
    :param samples_total: Samples in the complete fibonacci lattice, amount of returned points will be lower
    :param distance: max distance from tilt/azimuth
    :return: [tilts], [azimuths]
    """

    tilt_rad = numpy.radians(tilt)
    azimuth_rad = numpy.radians(azimuth)

    # xyz of tilt and azimuth
    x = math.cos(azimuth_rad) * math.sin(tilt_rad)
    y = math.sin(azimuth_rad) * math.sin(tilt_rad)
    z = math.cos(tilt_rad)


    print("xyz" + str(x) + " - " + str(y) + " " + str(z))


    # x y and z values for matplotlib test plotting
    xvals, yvals, zvals = [], [], []

    # actual tilt and azimuth values
    phis, thetas = [], []

    # doubling sample count as negative half of sphere is not needed
    # this should result in
    iterations = samples_total * 2
    for i in range(iterations):

        # using helper to get 5 values, x,y,z,phi,theta
        values = __get_fibonacci_sample(i, iterations)

        # if z is > 0, skip this loop iteration as bottom half of sphere is not needed
        if values[2] < 0:
            continue

        # xyz of fibonacci lattice points
        x_f = values[0]
        y_f = values[1]
        z_f = values[2]
        x_delta = x - x_f
        y_delta = y - y_f
        z_delta = z - z_f

        fibo_distance = math.sqrt(x_delta**2 + y_delta**2 + z_delta**2)

        print("fibonacci point distance: "  + str(fibo_distance))

        if fibo_distance < distance:
            # point was closer to tilt and azimuth than distance, can be added to outputs
            # add values to lists
            xvals.append(values[0])
            yvals.append(values[1])
            zvals.append(values[2])
            phis.append(values[3])
            thetas.append(values[4])

    print(xvals)
    print(yvals)
    print(zvals)
    # test plotting
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(xvals, yvals, zvals)
    matplotlib.pyplot.show()

    # returning tilt and azimuth values
    return phis, thetas






def get_fibonacci_distribution_tilts_azimuths(samples):
    """
    Returns tilt and azimuth values for the upper half of a fibonacci half sphere
    :param samples: approximate count for fibonacci half sphere points
    :return: [tilts(rad)], [azimuths(rad)], len(tilts) ~ samples
    """

    # x y and z values for matplotlib test plotting
    xvals, yvals, zvals = [], [], []

    # actual tilt and azimuth values
    phis, thetas = [], []

    # doubling sample count as negative half of sphere is not needed
    # this sould result in
    iterations = samples * 2
    for i in range(iterations):

        # using helper to get 5 values
        values = __get_fibonacci_sample(i, iterations)

        # if z is > 0, skip this loop iteration as bottom half of sphere is not needed
        if values[2] < 0:
            continue

        # add values to lists
        xvals.append(values[0])
        yvals.append(values[1])
        zvals.append(values[2])
        phis.append(values[3])
        thetas.append(values[4])

    # test plotting, shows the points in 3d space. Can be used for verification
    #fig = matplotlib.pyplot.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter3D(xvals, yvals, zvals)
    #matplotlib.pyplot.show()

    # returning tilt and azimuth values
    return phis, thetas


############################
#   HELPERS BELOW, CALL ONLY FROM WITHIN THIS FILE
############################

def __get_fibonacci_sample(sample, sample_max):
    """
    :param sample: sample number when there are sample_max samples
    :param sample_max: highest sample number
    :return: x(-1,1), y(-1, 1), z(-1,1), tilt(rad) and azimuth(rad) values for a single point on a fibonacci sphere
    """

    # Code based on sample at https://medium.com/@vagnerseibert/distributing-points-on-a-sphere-6b593cc05b42

    k = sample + 0.5

    # degrees from top to bottom in radians
    phi = math.acos(1 - 2 * k / sample_max)
    # azimuth, goes super high superfast, this is why modulo is used to scale values down
    theta = math.pi * (1 + math.sqrt(5)) * k
    theta = theta % (math.pi * 2)

    x = math.cos(theta) * math.sin(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(phi)

    return x, y, z, phi, theta



def angle_distance_between_points(tilt1, azimuth1, tilt2, azimuth2):
    """
    Calculates the angular distance in degrees between two points in angle space
    :param tilt1: point 1 tilt angle in degrees
    :param azimuth1: point 1 azimuth angle in degrees
    :param tilt2: point 2 tilt angle in degrees
    :param azimuth2: point 2 azimuth angle in degrees
    :return: sphere center angle between the two points
    """
    tilt1_rad = numpy.radians(tilt1)
    azimuth1_rad = numpy.radians(azimuth1)
    tilt2_rad =numpy.radians(tilt2)
    azimuth2_rad = numpy.radians(azimuth2)

    #print("Computing angular distance between two angle space points...")

    x1 = math.sin(tilt1_rad)*math.cos(azimuth1_rad)
    y1 = math.sin(tilt1_rad)*math.sin(azimuth1_rad)
    z1 = math.cos(tilt1_rad)

    #print("Point 1 x,y,z: " + str(round(x1, 2)) + " " + str(round(y1, 2)) + " " + str(round(z1,2)))

    x2 = math.sin(tilt2_rad) * math.cos(azimuth2_rad)
    y2 = math.sin(tilt2_rad) * math.sin(azimuth2_rad)
    z2 = math.cos(tilt2_rad)

    #print("Point 2 x,y,z: " + str(round(x2, 2)) + " " + str(round(y2, 2)) + " " + str(round(z2, 2)))

    euclidean_distance = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    center_angle = numpy.degrees(math.acos((2-euclidean_distance**2)/2))

    #print("Euclidean distance was: " + str(round(euclidean_distance, 4)) + " , angle delta: " +  str(round(center_angle, 4)) + " degrees")

    return center_angle



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

    #print("computing delta between measurements and poa simulation")
    #print(xa_day)
    #print(poa)

    # TODO this loop here is likely to be the main cause for angle estimation being slow
    # Replace with vectorized operations?
    for i in range(len(xa_minutes)):
        xa_minute = xa_minutes[i]  # this is poa index
        xa_power = xa_powers[i]
        poa_power = poa_powers[xa_minute]

        delta = xa_power - poa_power

        # poa power may be 0, avoiding zero divisions here
        if poa_power > 1:
            percent_delta = (delta / poa_power) * 100
        else:
            percent_delta = None
        deltas.append(delta)
        percent_deltas.append(percent_delta)
        minutes.append(xa_minute)

    return deltas, percent_deltas, minutes
