import math

import numpy

import angler
import polarplotter
import pymapper.mapper
import cloud_free_day_finder
import geoguesser_latitude
import geoguesser_longitude
import multiplier_matcher
import pvlib_poa
import solar_power_data_loader
import splitters
import matplotlib
import matplotlib.pyplot
import config
import time
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)


####################################
# Tests for data loading
####################################


def test_data_loading():  # works
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()


####################################
# Tests for data plotting
####################################


def plot_cloudy_day():
    ###############################################################
    #   This function plots a known cloudy day
    ###############################################################
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    # day 174 from year 2017 is known to be cloudy
    year_n = 2018
    day_n = 128
    day = splitters.slice_xa(data, year_n, year_n, day_n, day_n)

    # dropping nan values
    day = day.dropna(dim="minute")
    matplotlib.rcParams.update({'font.size': 18})

    # loading values as lists for plotting
    powers = day["power"].values[0][0]  # using[0][0] because the default output is [[[1,2,3,...]]] Might be due to how
    # xarray handles data variables with multiple coordinates
    minutes = day["minute"].values

    # plotting day and significant minutes
    matplotlib.pyplot.scatter(minutes, powers, s=[0.2] * len(minutes), c="black")

    matplotlib.pyplot.scatter([minutes[0]], [powers[0]], s=[30], c=config.ORANGE)
    matplotlib.pyplot.scatter(minutes[len(minutes) - 1], powers[len(minutes) - 1], s=[30], c=config.PURPLE)

    # adding legend and labels, showing plot
    # matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('Minute')
    matplotlib.pyplot.ylabel('Power')
    matplotlib.pyplot.title("Cloudy day [" + str(year_n) + "-" + str(day_n) + "]")
    matplotlib.pyplot.show()


def plot_clear_day():
    ###############################################################
    #   This function automatically detects and plots a clear day
    ###############################################################
    # loading data
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    # selecting year
    year_n = 2018

    # taking year from data
    year_data = splitters.slice_xa(data, year_n, year_n, 10, 350)

    # using a smoothness based method for detecting cloud free days
    clear_days = cloud_free_day_finder.find_smooth_days_xa(year_data, 70, 250, 0.5)

    # setting up plot
    matplotlib.rcParams.update({'font.size': 18})

    # day xa
    day = clear_days[0]

    # day number
    day_n = day["day"].values[0]

    # day with no nans
    day = day.dropna(dim="minute")

    # power values and minutes
    powers = day["power"].values[0][0]
    minutes = day["minute"].values

    # plotting measured values
    matplotlib.pyplot.scatter(minutes, powers, s=[0.2] * len(minutes), c="black")

    # plotting endpoints
    matplotlib.pyplot.scatter([minutes[0]], [powers[0]], s=[30], c=config.ORANGE)
    matplotlib.pyplot.scatter(minutes[len(minutes) - 1], powers[len(minutes) - 1], s=[30], c=config.PURPLE)

    # matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('Minute')
    matplotlib.pyplot.ylabel('Power')
    matplotlib.pyplot.title("Clear day [" + str(year_n) + "-" + str(day_n) + "]")

    matplotlib.pyplot.show()

def plot_year_of_data():
    data = solar_power_data_loader.get_fmi_kuopio_data_as_xarray()

    year_n = 2020



    days = []
    minutes = []
    powers = []


    for d in range(190, 300):
        d_data = splitters.slice_xa(data, year_n, year_n, d, d)
        d_data = d_data.dropna(dim="minute")
        d_powers = d_data["power"].values[0][0]  # using[0][0] because the default output is [[[1,2,3,...]]] Might be due to how
        # xarray handles data variables with multiple coordinates
        d_minutes = d_data["minute"].values
        d_n_day = d_data["day"].values
        #print("########")
        #print(len(powers))
        #print(len(minutes))

        day_ns = [d_n_day[0] for i in range(len(d_powers))]


        days.extend(day_ns)
        minutes.extend(d_minutes)
        powers.extend(d_powers)


    print("DAYS")
    #print(days)
    print(len(days))
    print("MINUTES")
    #print(minutes)
    print(len(minutes))
    print("POWERS")
    #print(powers)
    print(len(powers))


    fig = matplotlib.pyplot.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')


    # plotting day and significant minutes
    ax.scatter(minutes, powers, days, s=[0.2] * len(minutes), c="black")
    matplotlib.pyplot.show()

    #matplotlib.pyplot.scatter([minutes[0]], [powers[0]], s=[30], c=config.ORANGE)
    #matplotlib.pyplot.scatter(minutes[len(minutes) - 1], powers[len(minutes) - 1], s=[30], c=config.PURPLE)




####################################
# Tests for geolocation estimation
####################################


def estimate_longitude_v2():  # works, used in thesis
    ###############################################################
    #   This function contains an example on how to estimate geographic longitudes
    #   The method relies on geoguesser_longitude.estimate_longitude_based_on_year(year_data)
    #   Which has hardcoded geolocation for simulated solar noon time. The hardcoded values should be adjusted
    #   if the algorithm is to be used for datasets describing solar pv generation outside of Finland
    ###############################################################

    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    # selecting year
    year_n = 2018

    # taking year from data
    year_data = splitters.slice_xa(data, year_n, year_n, 125, 250)

    # estimating longitude with one year of data
    geoguesser_longitude.estimate_longitude_based_on_year(year_data)

    # estimating every year
    for year_n in range(2017, 2022):
        year_data = splitters.slice_xa(data, year_n, year_n, 125, 250)

        estimated_longitude = geoguesser_longitude.estimate_longitude_based_on_year(year_data)
        print("year: " + str(year_n) + " estimated longitude: " + str(estimated_longitude))



def estimate_latitude():  # works,  used in thesis
    ###############################################################
    #   This function contains an example on latitude estimation.
    #   Estimation uses functions from file geoguesser_latitude.py which is fairly messy
    ###############################################################
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    for year_n in range(2017, 2022):
        estimated_latitude = geoguesser_latitude.slopematch_estimate_latitude_using_single_year(data, year_n, 190, 280) # was 190-250
        print(
            "year " + str(year_n) + " lat 1: " + str(estimated_latitude[0]) + ", lat 2: " + str(estimated_latitude[1]))


####################################
# Test for multiplier estimation
####################################

def match_multiplier():  # not used in thesis, yet
    ###############################################################
    #   This function plots a clear day from dataset and a poa plot with automated multiplier matching
    ###############################################################
    # loading data
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    # selecting year
    year_n = 2018

    # taking year from data
    year_data = splitters.slice_xa(data, year_n, year_n, 10, 350)

    # using a smoothness based method for detecting cloud free days
    clear_days = cloud_free_day_finder.find_smooth_days_xa(year_data, 70, 250, 0.5)

    # setting up plot
    matplotlib.rcParams.update({'font.size': 18})

    # day xa
    day = clear_days[0]
    # day number
    day_n = day["day"].values[0]

    # creating single segment multiplier matched poa
    poa = pvlib_poa.get_irradiance(year_n, config.HELSINKI_KUMPULA_LATITUDE, config.HELSINKI_KUMPULA_LONGITUDE, day_n,
                                   15, 135)
    multiplier = multiplier_matcher.get_estimated_multiplier_for_day(day, poa)
    poa = pvlib_poa.get_irradiance_with_multiplier(year_n, config.HELSINKI_KUMPULA_LATITUDE,
                                                   config.HELSINKI_KUMPULA_LONGITUDE,
                                                   day_n, 15, 135, multiplier)

    # plotting single segment multiplier matched poa
    matplotlib.pyplot.plot(poa.minute.values, poa.POA.values, c=config.ORANGE, label="Area matched multiplier")

    # creating multi-segment multiplier matched poa
    poa2 = pvlib_poa.get_irradiance(year_n, config.HELSINKI_KUMPULA_LATITUDE, config.HELSINKI_KUMPULA_LONGITUDE, day_n,
                                    15, 135)
    multiplier = multiplier_matcher.get_estimated_multiplier_for_day_with_segments(day, poa2, 10)
    poa2 = pvlib_poa.get_irradiance_with_multiplier(year_n, config.HELSINKI_KUMPULA_LATITUDE,
                                                    config.HELSINKI_KUMPULA_LONGITUDE,
                                                    day_n, 15, 135, multiplier)

    # plotting multi-segment multiplier matched poa
    matplotlib.pyplot.plot(poa2.minute.values, poa2.POA.values, c=config.PURPLE, label="Segment matched multiplier")

    day = day.dropna(dim="minute")
    powers = day["power"].values[0][0]
    minutes = day["minute"].values

    # plotting scatter of measurements and significant points
    matplotlib.pyplot.scatter(minutes, powers, s=[0.2] * len(minutes), c="black")

    # plotting segment matched poa curve

    # matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel('Minute')
    matplotlib.pyplot.ylabel('Power')
    matplotlib.pyplot.legend()

    matplotlib.pyplot.show()


def plot_multi_segment_partly_cloudy():
    """
# plots n segment based multiplier matching visualization for a cloudy day
#:return:
    """
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    # selecting year
    year_n = 2018

    # known partly cloudy day
    day = splitters.slice_xa(data, year_n, year_n, 85, 85)

    matplotlib.rcParams.update({'font.size': 14})

    # segments = multiplier_matcher.get_measurements_split_into_n_segments(day, 10)

    poa = pvlib_poa.get_irradiance(year_n, config.HELSINKI_KUMPULA_LATITUDE, config.HELSINKI_KUMPULA_LONGITUDE, 85, 15,
                                   135)
    segments, multipliers2 = multiplier_matcher.get_segments_and_multipliers(day, 10, poa)
    print(multipliers2)

    matplotlib.pyplot.title("Day [2018 - 85]")
    matplotlib.pyplot.xlabel("Minute")
    matplotlib.pyplot.ylabel("Power(W)")

    for segment in segments:
        minutes = segment.minute.values
        powers = segment.power.values[0][0]
        matplotlib.pyplot.fill_between(minutes, powers, alpha=0.8)

    matplotlib.pyplot.show()


def plot_multi_year_geolocations_on_map():

    # Estimates geolocations for installation per year and plots them with pymapper


    #data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    #change site here, map and data loading will be adjusted accordingly
    site = "Helsinki"

    data = ""
    correct_long = 0.0
    correct_lat = 0.0
    if site == "Kuopio":
        data = solar_power_data_loader.get_fmi_kuopio_data_as_xarray()
        correct_lat = config.KUOPIO_FMI_LATITUDE
        correct_long = config.KUOPIO_FMI_LONGITUDE
    elif site == "Helsinki":
        data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()
        correct_lat = config.HELSINKI_KUMPULA_LATITUDE
        correct_long = config.HELSINKI_KUMPULA_LONGITUDE


    pymapper.mapper.toggle_grid(True)
    pymapper.mapper.create_map()

    # estimating every year
    for year_n in range(2017, 2022):

        fday = 190
        lday = 280
        #estimating longitude
        year_data = splitters.slice_xa(data, year_n, year_n, fday, lday)
        estimated_longitude = geoguesser_longitude.estimate_longitude_based_on_year(year_data)
        #print("year: " + str(year_n) + " estimated longitude: " + str(estimated_longitude))

        # estimating latitude
        estimated_latitude = geoguesser_latitude.slopematch_estimate_latitude_using_single_year(data, year_n, fday, lday)
        #print( "year " + str(year_n) + " lat 1: " + str(estimated_latitude[0]) + ", lat 2: " + str(estimated_latitude[1]))

        #taking the average of two estimated latitude values
        estimated_latitude_n = (estimated_latitude[0]+estimated_latitude[1])/2

        print("year " + str(year_n) + " predicted average: " + str(estimated_latitude_n))

        # plotting point and year on map
        pymapper.mapper.plot_point(estimated_latitude_n, estimated_longitude)
        pymapper.mapper.add_text_to_map(str(year_n), estimated_latitude_n, estimated_longitude)

    if site == "Helsinki":
        pymapper.mapper.limit_map_to_region(62.5, 59.2, 22, 28)
    elif site == "Kuopio":
        pymapper.mapper.limit_map_to_region(63.75, 60.2, 24, 31)



    pymapper.mapper.plot_point(correct_lat, correct_long, color="purple")

    pymapper.mapper.add_text_to_map("FMI " + site, correct_lat, correct_long)

    pymapper.mapper.show_map()



def test_one_panel_angle():
    ###############################################################
    #   Calculates a fitness value for a single panel angle pair
    ###############################################################
    print("evaluating one panel angle pair")
    # loading data
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    # selecting year
    year_n = 2018

    # taking year from data
    year_data = splitters.slice_xa(data, year_n, year_n, 10, 350)

    print("finding clear day")
    # using a smoothness based method for detecting cloud free days
    clear_days = cloud_free_day_finder.find_smooth_days_xa(year_data, 70, 250, 0.5)

    # day xa
    day = clear_days[0]
    day_n = day["day"].values[0]

    panel_tilt = 24.1
    panel_azimuth = 138.4

    print("computing panel angle fitness")

    fitness = angler.test_single_pair_of_angles(day, config.HELSINKI_KUMPULA_LATITUDE,
                                                  config.HELSINKI_KUMPULA_LONGITUDE, panel_tilt, panel_azimuth)

    print("fitness:")
    print(fitness)

    # PLOTTING TESTED POA

    powers = day["power"].values[0][0]
    minutes = day["minute"].values

    matplotlib.pyplot.scatter(minutes, powers, s=[0.2] * len(minutes), c="black")

    # poa
    best_tilt = panel_tilt
    best_azimuth = panel_azimuth
    best_poa = pvlib_poa.get_irradiance(year_n, config.HELSINKI_KUMPULA_LATITUDE, config.HELSINKI_KUMPULA_LONGITUDE,
                                        day_n, best_tilt, best_azimuth)
    multiplier = multiplier_matcher.get_estimated_multiplier_for_day(day, best_poa)

    matplotlib.pyplot.scatter(best_poa.minute.values, best_poa.POA.values * multiplier, s=0.2, c=config.ORANGE)

    matplotlib.pyplot.show()


def test_grid_of_panel_angles():
    ###############################################################
    #   Calculates fitness values for multiple panel angle pairs in a grid.
    ###############################################################
    # loading data
    data = solar_power_data_loader.get_fmi_kuopio_data_as_xarray()

    # selecting year
    year_n = 2018

    # taking year from data
    year_data = splitters.slice_xa(data, year_n, year_n, 10, 350)

    # using a smoothness based method for detecting cloud free days
    clear_days = cloud_free_day_finder.find_smooth_days_xa(year_data, 70, 250, 0.5)

    # day xa
    day = clear_days[0]

    tilts = []
    facings = []
    fitnesses = []

    for tilt in range(0, 90, 20):
        for facing in range(0, 360, 20):
            fitness = angler.test_single_pair_of_angles(day, config.HELSINKI_KUMPULA_LATITUDE,
                                                          config.HELSINKI_KUMPULA_LONGITUDE, tilt, facing)

            tilts.append(tilt)
            facings.append(facing)
            fitnesses.append(fitness)

            print("tilt: " + str(tilt) + " facing: " + str(facing) + " fitness: " + str(fitness))

    polarplotter.plot_polar_scattermap(tilts, facings, fitnesses)


def test_fibonacci_grid_of_panel_angles():
    ###############################################################
    #   Calculates fitness values for single day and multiple panel angle pairs in a grid.
    ###############################################################
    # loading data and setting parameters
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()
    expected_tilt = 15
    expected_azimuth = 135
    latitude = config.HELSINKI_KUMPULA_LATITUDE
    longitude = config.HELSINKI_KUMPULA_LONGITUDE



    # selecting year
    year_n = 2018

    # taking year from data
    year_data = splitters.slice_xa(data, year_n, year_n, 10, 350)

    # using a smoothness based method for detecting cloud free days
    clear_days = cloud_free_day_finder.find_smooth_days_xa(year_data, 70, 250, 0.5)

    # day xa
    day = clear_days[4]
    day_n = day["day"].values[0]
    print(day_n)

    # tilt, azimuth and their fitness
    tilts_rad, azimuths_rad, fits = angler.test_n_fibonacchi_sample_fitnesses_against_day(day, 1000, latitude, longitude)
    tilts_deg = numpy.degrees(tilts_rad)
    azimuths_deg = numpy.degrees(azimuths_rad)

    # solving best tilt and azimuth for later plotting
    best_tilt_deg = 0
    best_azimuth_deg = 0
    best_fit = math.inf
    for i in range(len(fits)):
        fit = fits[i]
        tilt = tilts_deg[i]
        azimuth = azimuths_deg[i]
        if fit < best_fit:
            best_fit = fit
            best_tilt_deg = tilt
            best_azimuth_deg = azimuth


    print("Estimation error calculation ##############")
    angler.angle_distance_between_points(expected_tilt, expected_azimuth, best_tilt_deg, best_azimuth_deg)
    print("###########################################")



    # block for printing out angles in latex table format
    """
    for i in range(len(fits)):
        print("\n")
        print("tilt: " + str( round(tilts_deg[i],2)))
        print("azimuth: " + str(round(azimuths_deg[i],2)))
        print("fit: " + str(round(fits[i],2)))


    for i in range(len(fits)):
        print(str(round(tilts_deg[i],2)) + " & " + str(round(azimuths_deg[i],2)) + " & " + str(round(fits[i],2)) + "\\\\")
    """

    # plotting map, this solves best fit internally so no need to pass it on
    polarplotter.plot_polar_scattermap(tilts_deg, azimuths_deg, fits, use_cmap=True)


    # finally plotting curves to compare
    powers = day["power"].values[0][0]
    minutes = day["minute"].values

    matplotlib.pyplot.plot(minutes, powers, c="black", label="Measurements")

    #poa
    best_poa = pvlib_poa.get_irradiance(year_n, config.HELSINKI_KUMPULA_LATITUDE, config.HELSINKI_KUMPULA_LONGITUDE, day_n, best_tilt_deg, best_azimuth_deg)
    multiplier = multiplier_matcher.get_estimated_multiplier_for_day(day, best_poa)
    matplotlib.pyplot.plot(best_poa.minute.values, best_poa.POA.values*multiplier, color=config.ORANGE, label="Best found fit")

    #poa normal
    best_poa = pvlib_poa.get_irradiance(year_n, config.HELSINKI_KUMPULA_LATITUDE, config.HELSINKI_KUMPULA_LONGITUDE,
                                        day_n, expected_tilt, expected_azimuth) # 15 135 helsinki, 15 217 kuopio
    multiplier = multiplier_matcher.get_estimated_multiplier_for_day(day, best_poa)
    matplotlib.pyplot.plot(best_poa.minute.values, best_poa.POA.values * multiplier, color=config.PURPLE,
                           label="Simulated values with known parameters")


    matplotlib.pyplot.legend()

    matplotlib.pyplot.show()


def test_fibonacci_grid_of_panel_angles_multiday(samples):
    ###############################################################
    #   Calculates fitness values for multiple days and multiple panel angle pairs in a fibonacci lattice
    ###############################################################

    start_time = time.time()
    # loading data
    data = solar_power_data_loader.get_fmi_kuopio_data_as_xarray()
    expected_tilt = 15
    expected_azimuth = 217
    latitude = config.KUOPIO_FMI_LATITUDE
    longitude = config.KUOPIO_FMI_LONGITUDE

    print("Data loaded at %s " % (time.time() - start_time))

    # creating list for clear days
    clear_days = []


    first_year = 2016
    last_year = 2020
    # looping through years and adding clear days to list
    for year_nl in range(first_year, last_year+1):
        print("taking days from year " + str(year_nl))
        year_data = splitters.slice_xa(data, year_nl, year_nl, 10, 350)
        cloud_frees = cloud_free_day_finder.find_smooth_days_xa(year_data, 140, 220, 0.5)
        clear_days.extend(cloud_frees)

    print("Cloud free days at %s " % (time.time() - start_time))

    # creating result lists
    best_tilts = []
    best_azimuths = []
    best_fits = []
    day_ns = []

    print("there are " + str(len(clear_days)) + " days in this test ")

    # computing results for each clear day
    for day in clear_days:
        day_n = day["day"].values[0]
        print("operating on day: " + str(day_n))

        # getting fitness values for each evaluation point
        tilts_rad, azimuths_rad, fits = angler.test_n_fibonacchi_sample_fitnesses_against_day(day, samples,
                                                                                              latitude,
                                                                                              longitude)
        # solving for best fit out of all points
        best_tilt, best_azimuth, best_fit = polarplotter.__get_best_fitness_out_of_results(tilts_rad, azimuths_rad, fits)

        # adding best fit to best fit lists
        best_tilts.append(best_tilt)
        best_azimuths.append(best_azimuth)
        best_fits.append(best_fit)
        day_ns.append(day_n)

    print("Evaluation done at %s " % (time.time() - start_time))

    print("best tilts:")
    print(numpy.degrees(best_tilts))
    print("best azimuths:")
    print(numpy.degrees(best_azimuths))
    print("best fits")
    print(best_fits)

    print("Angle space distances")

    for i in range(len(day_ns)):
        day = day_ns[i]
        tilt_rad = best_tilts[i]
        azimuth_rad = best_azimuths[i]
        tilt_deg = numpy.degrees(tilt_rad)
        azimuth_deg = numpy.degrees(azimuth_rad)

        delta_degrees = angler.angle_distance_between_points(expected_tilt, expected_azimuth, tilt_deg, azimuth_deg)

        print("day " + str(day) + " predicted " + str(round(tilt_deg,2)) + " " + str(round(azimuth_deg,2)) + " delta degrees: " + str(round(delta_degrees,2)))


    polarplotter.plot_polar_scattermap_points_with_texts(numpy.degrees(best_tilts), numpy.degrees(best_azimuths), day_ns)

def test_localized_lattice():
    angler.get_fibonacci_distribution_tilts_azimuths_near_coordinate(15, 135, 10000, 0.2)

#test_one_panel_angle()
#test_fibonacci_grid_of_panel_angles()
#test_localized_lattice()

plot_multi_year_geolocations_on_map()
#estimate_latitude()

#plot_year_of_data()
#test_fibonacci_grid_of_panel_angles_multiday(10000)