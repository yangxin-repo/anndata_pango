
import pandas as pd

# transform the dtype in pd.DataFrame to save
def transform_dataframe_dtype(dataframe: pd.DataFrame):
    # transform the nan value to "None" in str cols
    for col in dataframe.columns[dataframe.isna().any()].tolist():
        if pd.api.types.is_string_dtype(dataframe[col]):
            dataframe[col] = dataframe[col].fillna("None")

    # transform the index to object
    dataframe.index.name = None
    try:
        if not pd.api.types.is_any_real_numeric_dtype(dataframe.index):
            dataframe.index = dataframe.index.astype(object)
    except:
        pass

    # transform the str cols to object
    for col in dataframe.select_dtypes(exclude=['number', "bool", "category"]).columns:
        dataframe[col] = dataframe[col].astype(object)

    # trandform the category of str to object
    for col in dataframe.select_dtypes(include=["category"]).columns:
        if dataframe[col].cat.categories.dtype == "string":
            dataframe[col] = dataframe[col].astype(object)

    return dataframe