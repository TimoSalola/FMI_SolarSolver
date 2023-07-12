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

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)


####################################
# Tests for data loading
####################################

def test_data_loading():  # works
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()


####################################
# Tests for cloudy and cloud free day plotting
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


####################################
# Tests for geolocation estimation
####################################

def estimate_longitude_v2(): # works, used in thesis
    ###############################################################
    #   This function contains an example on how to estimate geographic longitudes
    #   The method relies on geoguesser_longitude_v2.estimate_longitude_based_on_year(year_data)
    #   Which has hardcoded geolocation for simulated solar noon time. The hardcoded values should be adjusted
    #   if the algorithm is to be used for datasets describing solar pv generation outside of Finland
    ###############################################################

    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    # selecting year
    year_n = 2018

    # taking year from data
    year_data = splitters.slice_xa(data, year_n, year_n, 125, 250)

    # estimating longitude with one year of data
    geoguesser_longitude_v2.estimate_longitude_based_on_year(year_data)

    # estimating every year
    for year_n in range(2017, 2022):
        year_data = splitters.slice_xa(data, year_n, year_n, 125, 250)

        estimated_longitude = geoguesser_longitude_v2.estimate_longitude_based_on_year(year_data)
        print("year: " + str(year_n) + " estimated longitude: " + str(estimated_longitude))



def estimate_latitude(): # works,  used in thesis
    ###############################################################
    #   This function contains an example on latitude estimation.
    #   Estimation uses functions from file geoguesser_latitude.py which is fairly messy
    ###############################################################
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    for year in range(2017, 2022):
        estimated_latitude = geoguesser_latitude.slopematch_estimate_latitude_using_single_year(data, year, 190, 250)
        print("year " + str(year) + " lat 1: " + str(estimated_latitude[0]) + ", lat 2: " + str(estimated_latitude[1]))

####################################
# Test for multiplier estimation
####################################


def match_multiplier(): # not used in thesis, yet
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
    matplotlib.pyplot.plot(poa.minute.values, poa.POA.values,  c=config.ORANGE, label="Area matched multiplier")

    # creating multi-segment multiplier matched poa
    poa2 = pvlib_poa.get_irradiance(year_n, config.HELSINKI_KUMPULA_LATITUDE, config.HELSINKI_KUMPULA_LONGITUDE, day_n,
                                   15, 135)
    multiplier = multiplier_matcher.get_estimated_multiplier_for_day_with_segments(day, poa2,10)
    poa2 = pvlib_poa.get_irradiance_with_multiplier(year_n, config.HELSINKI_KUMPULA_LATITUDE,
                                                   config.HELSINKI_KUMPULA_LONGITUDE,
                                                   day_n, 15, 135, multiplier)

    # plotting multi-segment multiplier matched poa
    matplotlib.pyplot.plot(poa2.minute.values, poa2.POA.values,  c=config.PURPLE, label="Segment matched multiplier")

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
    data = solar_power_data_loader.get_fmi_helsinki_data_as_xarray()

    # selecting year
    year_n = 2018

    # known partly cloudy day
    day = splitters.slice_xa(data, year_n, year_n, 85, 85)

    matplotlib.rcParams.update({'font.size': 14})

    #segments = multiplier_matcher.get_measurements_split_into_n_segments(day, 10)

    poa = pvlib_poa.get_irradiance(year_n, config.HELSINKI_KUMPULA_LATITUDE, config.HELSINKI_KUMPULA_LONGITUDE,85, 15, 135)
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

plot_multi_segment_partly_cloudy()