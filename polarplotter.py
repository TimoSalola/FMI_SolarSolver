import math

import matplotlib
import numpy
from mpl_toolkits.mplot3d import Axes3D

import config
import multiplier_matcher
import pvlib_poa


def plot_polar_scattermap(tilts_deg, azimuths_deg, fitnessess, use_cmap = False):

    """
    Plots a polar scattermap of given tilt and azimuth angles with best fit marked. Uses cmap if (use_cmap) = true
    :param tilts_deg: list of tilts
    :param azimuths_deg: list of degrees
    :param fitnessess: list of fitness values, lower is better
    :param use_cmap: optional, use cmap or not
    :return: None
    """
    # translating tilts and azimuths from rads to degrees for plotting
    tilts_rad = numpy.radians(tilts_deg)
    azimuths_rad = numpy.radians(azimuths_deg)

    # taking logarithms of fitness value to help with plotting
    fitnesses_log = []
    for fit in fitnessess:
        fitnesses_log.append(math.log(fit))

    # Creating plot
    f = matplotlib.pyplot.figure(figsize=(13, 8))
    ax = matplotlib.pyplot.subplot(111, projection='polar')
    # setting zero at clock 12 and rotation as clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xlim([0.0, 2 * math.pi])  # angle limit from 0 to 2pi
    ax.set_ylim([0.0, 90.0])  # distance limit to 0 to 90 degrees

    # Scattering measurements and adding bar label

    if use_cmap:
        scatter = ax.scatter(azimuths_rad, tilts_deg, c=fitnesses_log, cmap="Greys_r")
    else:
        scatter = ax.scatter(azimuths_rad, tilts_deg, c="black")
    matplotlib.pyplot.colorbar(scatter, label="Log(Delta)")

    # solving best point and scattering it
    best_tilt, best_azimuth, best_fit = __get_best_fitness_out_of_results(tilts_deg, azimuths_deg, fitnesses_log)
    ax.scatter(numpy.radians(best_azimuth), best_tilt, marker="x", color="red", s=20)
    ax.text(numpy.radians(best_azimuth), best_tilt, "Best fit", c="red", size=20)

    print("Best tilt: " + str(best_tilt) + " best azimuth: " + str(best_azimuth) + " best fit: " + str(best_fit))

    #matplotlib.pyplot.title("Polar scattermap")
    matplotlib.pyplot.show()

    ### Optional plot day curve vs poa curve


def plot_polar_scattermap_points_with_texts(tilts_deg, azimuths_deg, days):
    tilts_rad = numpy.radians(tilts_deg)
    azimuths_rad = numpy.radians(azimuths_deg)

    print("tilts:")
    print(tilts_deg)
    print("azimuths:")
    print(azimuths_rad)


    # Creating plot
    f = matplotlib.pyplot.figure(figsize=(13, 8))
    ax = matplotlib.pyplot.subplot(111, projection='polar')
    # setting zero at clock 12 and rotation as clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xlim([0.0, 2 * math.pi])  # angle limit from 0 to 2pi
    ax.set_ylim([0.0, 90.0])  # distance limit to 0 to 90 degrees

    # Scattering measurements and adding bar label
    scatter = ax.scatter(azimuths_rad, tilts_deg, alpha= 0.2, color="black")

    # adding day numbers
    for i in range(len(days)):
        tilt = tilts_deg[i]
        azimuth = azimuths_rad[i]
        day = days[i]
        #ax.text(azimuth, tilt, day, c="red", size=20)

    #matplotlib.pyplot.colorbar(scatter, label="Log(Delta)")
    matplotlib.pyplot.show()





def __get_best_fitness_out_of_results(tilt_rads, azimuth_rads, fitnesses):
    """
    Returns tilt and azimuth values for lowest(best) fitness value
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