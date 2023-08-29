#####################################################################
#   Functions for multiplier matching
#   Only get_estimated_multiplier_for_day and __helpers are needed, other functions are
#####################################################################

def get_estimated_multiplier_for_day(measurements, poa):
    """
    Generates area based multiplier value
    :param measurements: xarray containing power measurements
    :param poa: dataframe from pvlib_poa containing simulated irradiance values
    :return: ratio between measurements and poa, can be used as a multiplier
    """

    # dropping nan values from measurements as nans in measurements cause nan as sum and thus invalid output
    measurements = measurements.dropna(dim="minute")

    measured_minutes, measured_powers = __measurements_to_mins_powers(measurements)
    poa_minutes, poa_powers = __poa_to_mins_powers(poa)

    sum_poa = sum(poa_powers)
    sum_mea = sum(measured_powers)

    if sum_poa == 0 or sum_mea == 0:
        return None

    ratio = sum_mea / sum_poa

    print("sum of poa " + str(sum_poa) + " and measurements " + str(sum_mea) + " ratio was " + str(ratio))
    return ratio


def get_measurement_segment_n_of_k(measurements, n, k):
    """
    Returns measurements which belong in the nth of k segment
    :param measurements: power measurements xarray
    :param n: nth segment, segment 4 of 8 for example
    :param k: total segment count, 8 for example
    :return: xarrya containing values within segment
    """

    # dropping leading and tailing nans
    measurements = measurements.dropna(dim="minute")
    minutes, powers = __measurements_to_mins_powers(measurements)

    first_min = minutes[0]
    last_min = minutes[len(minutes) - 1]

    print("first and last minutes: " + str(first_min) + " to " + str(last_min))

    segment_len = (last_min - first_min * 1.0) / k

    segment_start_min = int(first_min + (n - 1) * segment_len)
    segment_last_min = int(first_min + n * segment_len) - 1

    print("segment " + str(n) + " of " + str(k) + " was " + str(segment_start_min) + " to " + str(segment_last_min))

    measurements_in_range = measurements.where(measurements["minute"] >= segment_start_min)
    measurements_in_range = measurements_in_range.where(measurements_in_range["minute"] <= segment_last_min)
    measurements_in_range = measurements_in_range.dropna(dim="minute")

    return measurements_in_range


def get_measurements_split_into_n_segments(measurements, segment_count):
    """
    Splits measurements into n segments and returns segments as a list of xa
    :param measurements: xarray of power measurements
    :param segment_count: (int)amount of segments to split measurements into
    :return: [segment(xa), segment(xa), segment(xa)...]
    """

    segments = []

    for i in range(segment_count):
        segment = get_measurement_segment_n_of_k(measurements, i + 1, segment_count)
        segments.append(segment)

    return segments


def get_segments_and_multipliers(measurements, segment_count, poa):
    # splitting measurements into segments
    segments = get_measurements_split_into_n_segments(measurements, segment_count)

    # calculating multiplier for each segment
    multipliers = []

    for segment in segments:
        # print(segment)
        first_segment_minute = segment.minute.values[0]
        last_segment_minute = segment.minute.values[len(segment.minute.values) - 1]
        poa_in_range = poa.where(poa["minute"] >= first_segment_minute)
        poa_in_range = poa_in_range.where(poa["minute"] <= last_segment_minute)
        poa_in_range = poa_in_range.dropna()
        # print(poa_in_range)
        multiplier = get_estimated_multiplier_for_day(segment, poa_in_range)
        multipliers.append(multiplier)

    # returning segments and their multipliers

    return segments, multipliers


def get_cluster_multiplier_and_segments(segments, multipliers, percents):
    max_multiplier = max(multipliers)

    window_size = max_multiplier * (percents / 100)

    # print(window_size)

    best_range_count = 0
    best_range_center = 0

    for i in range(len(multipliers)):
        center = multipliers[i]
        low = center - window_size / 2
        high = center + window_size / 2
        in_range = sum(1 for value in multipliers if low <= value <= high)

        if in_range > best_range_count:
            best_range_count = in_range
            best_range_center = center

    # print("in range: " + str(best_range_count))
    # print("range: " + str(best_range_center-window_size/2) + " - " +str(best_range_center+window_size/2))

    segments_in_range = []
    multipliers_in_range = []
    for i in range(len(multipliers)):

        multiplier = multipliers[i]
        segment = segments[i]

        if best_range_center - window_size / 2 <= multiplier <= best_range_center + window_size / 2:
            segments_in_range.append(segment)
            multipliers_in_range.append(multiplier)

    average_multiplier = sum(multipliers_in_range) / len(multipliers_in_range)
    # print(multipliers_in_range)
    # print(segments_in_range)

    return average_multiplier, segments_in_range


def get_estimated_multiplier_for_day_with_segments(measurements, poa, segments):
    """
    Fairly complex function, splits the measurement and poa values into segments and calculates multiplier values for each segment
    :param measurements: XA with power generation measurements
    :param poa: pandas dataframe with POA simulations
    :param segments: segment count, 10 is a good default value
    :return: suggested multiplier value
    """

    # splitting measurements and poa to minutes and power values
    measured_minutes, measured_powers = __measurements_to_mins_powers(measurements)
    poa_minutes, poa_powers = __poa_to_mins_powers(poa)

    # figuring out the time interval as first and last minutes of measurements and poa are unlikely to align
    first_min = min(measured_minutes[0], poa_minutes[0])
    last_min = max(measured_minutes[len(measured_minutes) - 1], poa_minutes[len(poa_minutes) - 1])

    # calculating segment length
    segment_size = (last_min - first_min) / segments
    # print("segment size: " + str(segment_size))

    # starting point and end point for the first interval
    start = first_min
    end = start + segment_size

    # keeping a list of all the intervals and their ratios(multipliers)
    ends = []
    starts = []
    ratios = []

    # calculating multiplier for each segment
    for segment in range(segments):
        # print("segment " + str(segment)+ " interval " + str(start) + " to " + str(end))
        mea_segment = measurements.where((start <= measurements.minute) & (measurements.minute < end))
        mea_segment = mea_segment.dropna(dim="minute")

        poa_segment = poa.where((start <= poa.minute) & (poa.minute < end))
        poa_segment = poa_segment.dropna()

        ratio = get_estimated_multiplier_for_day(mea_segment, poa_segment)

        if ratio is None:
            # if ratio is faulty, don't add
            # ends.append(end)
            # starts.append(start)
            # ratios.append(0)

            start += segment_size
            end += segment_size
            continue

        ends.append(end)
        starts.append(start)
        ratios.append(ratio)

        start += segment_size
        end += segment_size

    # sorting ratios as that makes 1D clustering easier
    # print(ratios)
    sorted_ratios = ratios.copy()
    sorted_ratios.sort()

    # print(sorted_ratios)

    # accepted cluster range, this means that in array of 1, 2,3,4,5,6,6,6 ,7 the range of 0 would pick up the 6,6,6
    # segment while as range of 1 would pick up 5,6,6,6,7. In other words, how much the center value is allowed to
    # deviate for other values to still be able to be classified as values of the same cluster
    # adjust this value if you want the clustering interval to be tighter, for example 0.5 would be tighter where as 2
    # would choose a wider cluster
    cluster_range = 1

    # keeping up largest found cluster center points and first/last accepted value indexes
    highest_total_range = 0
    highest_total_range_min_index = 0
    highest_total_range_max_index = 0

    # reading values from sorted multiplier list one by one
    for center_index in range(len(sorted_ratios)):
        center_value = sorted_ratios[center_index]
        # print(center_index)

        in_negative_direction = 0
        in_positive_direction = 0

        # going backwards in the list and checking if multipliers could belong into the same cluster
        for i in range(1, len(sorted_ratios)):
            if center_index - i >= 0:
                value_at_i = sorted_ratios[center_index - i]
                if value_at_i + cluster_range >= center_value:
                    in_negative_direction += 1
                else:
                    break
            else:
                break

        # going forwards in the list and checking if multipliers could belong into the same cluster
        for i in range(1, len(sorted_ratios)):
            if center_index + i < len(sorted_ratios):
                value_at_i = sorted_ratios[center_index + i]
                if value_at_i - cluster_range <= center_value:
                    in_positive_direction += 1
                else:
                    break
            else:
                break

        # now that we know how far we could go, let's calculate interval length
        total_range = 1 + in_positive_direction + in_negative_direction

        # and update best found cluster
        if total_range > highest_total_range:
            highest_total_range = total_range
            highest_total_range_min_index = center_index - in_negative_direction
            highest_total_range_max_index = center_index + in_positive_direction

        # print("center " + str(center_value) + " downwards " + str(in_negative_direction) + " upwards" + str(
        # in_positive_direction))

    print("best interval was " + str(highest_total_range) + " from index " + str(highest_total_range_min_index) + " to "
                                                                                                                  "index " + str(
        highest_total_range_max_index))

    sum_of_in_range = 0
    for i in range(highest_total_range_min_index, highest_total_range_max_index + 1):
        sum_of_in_range += sorted_ratios[i]

    average = sum_of_in_range / highest_total_range

    print(average)

    return average


#### HELPERS

def __poa_to_mins_powers(poa):
    """
    :param poa: plane of array irradiance simulation data
    :return: minutes and corresponding power values
    """
    return poa.minute.values, poa.POA.values


def __measurements_to_mins_powers(mea):
    """
    :param mea: measurements xa
    :return: minutes and corresponding power values
    """
    return mea.minute.values, mea.power.values[0][0]
