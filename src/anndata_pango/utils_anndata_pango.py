
import pandas as pd
import h5py
import numpy as np
from anndata._io.specs.registry import _REGISTRY
from anndata.io import write_elem
from anndata._io.specs import IOSpec
from anndata_pango.image_pango.image_pango import Image_Pango

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

@_REGISTRY.register_write(
    h5py.Group,
    Image_Pango,
    IOSpec("image_pango", "0.1.0")
)
def write_image_pango(f, k, elem, dataset_kwargs=None):

    # If dataset_kwargs is None, initialize it as an empty dictionary
    if dataset_kwargs is None:
        dataset_kwargs = {}

    # Create a group for the Image_Pango object
    g = f.create_group(k)

    # Store pixel_matrix (numpy array)
    pixel_data = None
    if getattr(elem, "pixel_matrix", None) is not None:
        # If pixel_matrix exists, use it directly
        pixel_data = elem.pixel_matrix
    else:
        # If pixel_matrix does not exist, check if source_image exists
        if getattr(elem, "source_image", None) is not None:
            pixel_data = np.array(elem.source_image)

    write_elem(g, "pixel_matrix", pixel_data, dataset_kwargs=dataset_kwargs)

    # Store source_image (None)
    source_image_to_store = None
    write_elem(g,"source_image",source_image_to_store, dataset_kwargs=dataset_kwargs)
    
    # Store mask_matrix (dictionary, containing CSR sparse matrices, AnnData native support)
    write_elem(g, "mask_matrix", elem.mask_matrix, dataset_kwargs=dataset_kwargs)

    # Store coordinate (Coordinate_Pango object)
    coord = getattr(elem, "coordinate", None)
    if coord is not None:
        coord_g = g.create_group("coordinate")
        coord_dict = coord.__dict__

        # store source_coordinate
        if coord_dict.get("source_coordinate") is not None:
            write_elem(coord_g, "source_coordinate",
                       coord_dict["source_coordinate"], dataset_kwargs=dataset_kwargs)

        # store source_coordinate_mapping: DataFrame
        if coord_dict.get("affine_matrix") is not None:
            write_elem(coord_g, "affine_matrix",
                       coord_dict["affine_matrix"], dataset_kwargs=dataset_kwargs)

        # store coordinate_mapping: dict of DataFrames
        mapping = coord_dict.get("coordinate_mapping")
        if mapping is not None:
            mapping_g = coord_g.create_group("coordinate_mapping")
            for key, df in mapping.items():
                write_elem(mapping_g, key, df, dataset_kwargs=dataset_kwargs)

        coord_g.attrs["encoding-type"] = "coordinate_pango"
        coord_g.attrs["encoding-version"] = "0.1.0"

    return None
