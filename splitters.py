############################
#   FUNCTIONS FOR SPLITTING DATAFRAMES
############################

def slice_xa(xa, year_first, year_last, day_first, day_last):
    """
    THIS MALFUNCTIONS WITH MULTI YEAR DATAFRAMES
    """
    correct_year = xa.sel(year=slice(year_first, year_last))
    correct_year_correct_day = correct_year.sel(day=slice(day_first, day_last))

    return correct_year_correct_day


def slice_df(df, first_day, last_day):
    return df[(df["day"] >= first_day) & (df["day"] <= last_day)]


def slice_xa_minutes(xa, start_min, end_min):
    """
    Get minutes from start_min to end_min form dataframe
    """
    return xa.sel(minute=slice(start_min, end_min))


def get_xa_values_within_percentage_of_max_interval(xa_day, low, high):
    """
    :param xa_day: measurements from a single day
    :param low: low cutoff percent, for example 10 for values lower than 10% of max are rejected
    :param high: high cutoff percent, for example 90 for values high than 90% of max are rejected
    :return: xa_day with only values within specified range
    """
    xa_day = xa_day.dropna(dim="minute")
    highest_value = max(xa_day.power.values[0][0])

    high_cutoff = highest_value * (high / 100)
    low_cutoff = highest_value * (low / 100)
    # print("cutoffs: " + str(low_cutoff) + " and " + str(high_cutoff))

    xa_day = xa_day.where(xa_day.power >= low_cutoff)
    xa_day = xa_day.where(xa_day.power <= high_cutoff)

    xa_day = xa_day.dropna(dim="minute")
    return xa_day


def split_xa_to_3_lists(xa):
    """
    :param xa:
    :return: days, minutes, powers as 3 lists
    """

    days = []
    minutes = []
    powers = []

    xa_df = xa.to_dataframe()

    # print("xa_df size:" + str(len(xa_df)))
    # print("after dropna: " + str(len(xa_df.dropna())))

    xa_df = xa_df.dropna()

    year_zero = 0
    for year, df1 in xa_df.groupby("year"):
        if year_zero == 0:
            year_zero = year
        year_delta = year - year_zero
        for day, df2 in df1.droplevel("year").groupby("day"):
            for minute, df3 in df2.droplevel("day").groupby("minute"):
                days.append(day + year_delta * 365)
                minutes.append(minute)
                powers.append(df3["power"].values[0])
    return days, minutes, powers
