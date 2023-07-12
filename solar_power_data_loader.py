import pandas
import xarray


############################
#   FUNCTIONS FOR LOADING DATA
#   USE FIRST 3 FUNCTIONS, HELSINKI AND KUOPIO FMI CONTAIN HIGH QUALITY DATA
#   OTHER __FUNCTIONS ARE HELPERS, INTENDED FOR INTERNAL USE
############################


def get_fmi_helsinki_data_as_xarray():
    # filepath
    path = "fmi-helsinki-2021.csv"
    return __load_csv_as_xa(path)


def get_fmi_kuopio_data_as_xarray():
    # filepath
    path = "fmi-kuopio-2021.csv"
    return __load_csv_as_xa(path)


def get_oomi_laanila_oulu():
    path = "laanilan-koulu-oulu-2022.csv"
    data = __load_oomi_csv_as_xa(path)
    return data


def __load_oomi_csv_as_xa(csv_filename):
    """
    :param csv_filename: Name of csv file
    :return: xarray containing power generation data
    """


    """
    Function for processing another type of solar CSV files
    Files following this structure will not be included in the project
    In this format, power measurements are given every 10 minutes and in KWh -format
    This function transforms KWh per 10 min to W per s 
    """


    # expecting power to be in kwh over 10min, so 10 min to 60min = times 6
    # kwh to wh = 1000
    # 6000

    print("Loading oomidata from file " + str(csv_filename))
    data = pandas.read_csv(
        csv_filename,
        sep=";",  # ; normal separator
        skiprows=2,  # first 2 contain names
        names=["date", "power", "empty"],  # "data;data;empty"
        # nrows=10000000,
        encoding='unicode_escape',  # removes an unicode decoder error
        skipfooter=6,  # last 6 rows have averages and other data
        engine="python",  # removes a warning from console
        decimal=","  # seems to use 0,543 for power values
    )

    # loading dates and converting to datetime
    dates = data["date"]
    data["date"] = pandas.to_datetime(dates)

    # transforming kwh per 10min to w
    data["power"] = data["power"] * 6000  # for 10min kwh to w

    filtered_dataframe = pandas.DataFrame(data, columns=["date", "power"])

    minute_format_dataframe = __minute_format(filtered_dataframe)

    indexed_dataframe = minute_format_dataframe.set_index(["year", "day", "minute"])

    indexed_dataframe = indexed_dataframe[~indexed_dataframe.index.duplicated()]
    # print(indexed_dataframe)

    xa = indexed_dataframe.to_xarray()
    # print(xa)

    return xa


def __load_csv_as_xa(csv_filename):
    """
    WARNING, THIS REQUIRES THE CSV FILE TO FOLLOW SAME PATTERN AS FMI HELSINKI AND FMI KUOPIO
    :param csv_filename:
    :return:
    """

    """
    FMI data file pattern
    prod_time, pv_inv_out, pv_inv_in, pv_str_1, pv_str_2
    2015-08-26 03:34:00
    2015-08-26 03:35:00
    2015-08-26 03:36:00
    """

    print("Loading data from " + csv_filename)

    # importing data
    data = pandas.read_csv(
        csv_filename,
        sep=";",
        skiprows=16,
        names=["date", "output to grid", "power", "PV1", "PV2"],
        nrows=10000000
    )

    # dropping nan here might not be needed
    data = data.dropna()

    # modifying datetime field type to datetime.
    data["date"] = pandas.to_datetime(data["date"])

    # picking out the important fields of date and power from the dataframe
    filtered_dataframe = pandas.DataFrame(data, columns=["date", "power"])

    # changing format to our year, day minute, power -format. This splits the date to 3 fields
    df_minutes = __minute_format(filtered_dataframe)

    # naming the new 3 index columns
    df_minutes = df_minutes.set_index(["year", "day", "minute"])

    # printing how much data came out
    print("Read " + str(len(df_minutes)) + " rows.")

    # there should be some missing values, starting by dropping nans
    df_minutes = df_minutes.where(df_minutes.power > 0)
    df_minutes = df_minutes.dropna(how="any")  # any should remove values when year, day, minute or power are nan


    # interpolating nan values
    df_minutes = __fill_missing_values_df(df_minutes)
    # above adds nans for some reason? They have to be removed again

    # removing new nans
    df_minutes = df_minutes.dropna()

    # transforming dataframe to xarray
    xa = df_minutes.to_xarray()

    return xa


def __fill_missing_values_df(df):
    print("\tFilling missing values in df")

    '''
    This function is not especially pretty. 
    Dicts are used to iterate through the data and missing minutes are linearly approximated
    '''

    # creating a dict where year maps to a list of days for that year in dataset
    # should contain 2016 to [1, 2,3,3,4, ... 365], 2017 to [1,2...]
    year_to_day_list_dict = dict()

    # filling the year to day list dict
    for index_tuple in df.index.values:
        year = index_tuple[0]
        day = index_tuple[1]
        if year in year_to_day_list_dict:
            day_list = year_to_day_list_dict.get(year)
            if day_list[len(day_list) - 1] != day:
                day_list.append(day)
                year_to_day_list_dict[year] = day_list
        else:
            year_to_day_list_dict[year] = [day]


    full_set_of_years = []

    # iterating through the dict where key = year, value = days of that year
    for key in year_to_day_list_dict.keys():

        # using the year key to slice the dataframe so that we get only that specific year
        df_of_year = df.xs(key, level=0)

        # list of every single day corresponding to this specific year
        year_days = year_to_day_list_dict[key]

        # list days without missing minutes from this key year
        days_without_missing_minutes = []

        # Looping through each day in year_days
        for day in year_days:
            # slicing the specific day from year long dataframe
            day = df_of_year.xs(day, level=0)
            # filling missing minutes from the day
            day_without_missing_minutes = __fill_missing_minute_values_within_given_day(day)
            # adding the day without missing minutes to a list of multiple days without missing minutes
            days_without_missing_minutes.append(day_without_missing_minutes)

        # merging the lists of days together to create a full year
        year_df = pandas.concat(days_without_missing_minutes, keys=year_days, names=["day", "minute"])
        # adding this year_df containing days without missing minutes to a list of processed years
        full_set_of_years.append(year_df)

    # merging processed years together, this final df should not be missing any minutes
    output_df = pandas.concat(full_set_of_years, keys=year_to_day_list_dict.keys(), names=["year", "day", "minute"])
    print("\tMissed values are now filled")
    return output_df


def __fill_missing_minute_values_within_given_day(day):
    """
    Linear interpolation of missing minutes method
    :param day: pandas dataframe containing one day of data
    :return:
    """

    """
        Dataframe structure:
                power
        minute       
        296      14.0
        297      15.0
        298      25.0
        299      42.4
        300      40.1
    """

    first_minute = min(day.index)
    last_minute = max(day.index)

    missing_minutes = []
    missing_values = []

    # determines if trailing or leading values should be added
    fill_before_after = False

    for i in range(1440):
        if fill_before_after:
            if i < first_minute or i > last_minute:
                missing_minutes.append(i)
                missing_values.append(0)
            elif i not in day.index:
                missing_minutes.append(i)
                missing_values.append(float("nan"))
        else:
            if first_minute <= i <= last_minute and i not in day.index:
                missing_minutes.append(i)
                missing_values.append(float("nan"))

    missing_dict = {"minute": missing_minutes, "power": missing_values}
    df_missing_values = pandas.DataFrame.from_dict(missing_dict)
    df_missing_values = df_missing_values.set_index("minute")
    full1440day = pandas.concat([day, df_missing_values])
    full1440day = full1440day.sort_index()

    # INTERPOLATION HAPPENS HERE, interpolation is done twice, once from both different directions as the interpolation
    # method does not seem to be direction invariant. Using two interpolations seemed to fix the is
    full1440day = full1440day.interpolate(method="linear")
    full1440day = full1440day.reindex(index=full1440day.index[::-1])
    full1440day = full1440day.interpolate(method="linear")
    full1440day = full1440day.reindex(index=full1440day.index[::-1])

    return full1440day


def __minute_format(dataframe):
    """
    :param dataframe: datetime - power dataframe
    :return: year - day - minute - power dataframe
    """
    print("*Reformatting dataframe to [Year, Day, Minute, Power] -format")

    output = pandas.DataFrame.copy(dataframe, deep=True)

    output["minute"] = output["date"].dt.hour * 60 + output["date"].dt.minute

    output["year"] = output["date"].dt.year
    output["day"] = output["date"].dt.strftime("%j").astype(int)

    output2 = pandas.DataFrame(output, columns=["year", "day", "minute", "power"])

    return output2