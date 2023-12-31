import pandas as pd

def drop_duplicates_ctoi_tess(tess_data, ctoi_data):
    """Merge two datasets based on 'TIC ID' and return
    the rows present in ctoi_data but not in tess_data.

    Args:
        tess_data (DataFrame): The first dataset.
        ctoi_data (DataFrame): The second dataset.

    Returns:
        DataFrame: Subset of ctoi_data containing rows not present in tess_data.
    """
    merged_data = pd.merge(ctoi_data, tess_data[['TIC ID']], on='TIC ID', how='left', indicator=True)
    ctoi_data_removed = merged_data[merged_data['_merge'] == 'left_only'].drop(columns=['_merge'])
    return ctoi_data_removed
