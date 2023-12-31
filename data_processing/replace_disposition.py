from utils.mission import Mission

def replace_label(data, mission=Mission.KEPLER):
    """Replace labels in the data based on the specified mission.

    Args:
        data (DataFrame): The observation data.
        mission (str): The mission name.

    Returns:
        DataFrame: Labeled data.
        -1 if the mission name is invalid.
    """
    if mission == Mission.KEPLER:
        # Map the classes to label 0 and 1
        mapping = {'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}
        # Replace the values using the map
        data['koi_disposition'] = data['koi_disposition'].map(mapping)
        return data
    elif mission == Mission.TESS:
        mapping = {'KP':1,'CP':1,'PC':1,'APC':0,'FA':0,'FP':0}
        data['TFOPWG Disposition'] = data['TFOPWG Disposition'].map(mapping)
        return data
    elif mission == Mission.CTOI:
        mapping = {'PC': 1, 'FP': 0}
        data['User Disposition'] = data['User Disposition'].map(mapping)
        return data
    elif mission == Mission.K2:
        mapping = {'CONFIRMED': 1, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}
        data['disposition'] = data['disposition'].map(mapping)
        return data
    else:
        print('warning: invalid mission name.')
        return -1