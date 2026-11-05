
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
from anndata._io.specs import IOSpec
from anndata._io.specs.registry import _REGISTRY
from anndata.io import write_elem
from cell_type_mapper.cli.from_specified_markers import FromSpecifiedMarkersRunner
from pyensembl import EnsemblRelease, Genome

from anndata_pango.image_pango.image_pango import Image_Pango


# transform the dtype in pd.DataFrame to save
def transform_dataframe_dtype(dataframe: pd.DataFrame) -> pd.DataFrame:
    # transform the nan value to "None" in str cols
    for col in dataframe.columns[dataframe.isna().any()].tolist():
        if pd.api.types.is_string_dtype(dataframe[col]):
            dataframe[col] = dataframe[col].fillna("None")

    # transform the index to object
    dataframe.index.name = None
    if not pd.api.types.is_any_real_numeric_dtype(dataframe.index):
        dataframe.index = dataframe.index.astype(object)


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

def map_cell_type_pango(adata:ad.AnnData,
                        cache_dir:str | None=None,
                        query_markers_path:str | None=None,
                        precomputed_stats_path:str | None=None,
                        output_dir:str | None=None,
                        makers_name:Literal["mouse_markers_230821"]="mouse_markers_230821",
                        stats_name:Literal["precomputed_stats_ABC_revision_230821"]="precomputed_stats_ABC_revision_230821",
                        inplace:bool=True) -> ad.AnnData | None:

    # verify the markers and stats name
    makers_name = f"{makers_name}.json"
    stats_name = f"{stats_name}.h5"

    # verify the cache directionary
    if cache_dir is None:
        if query_markers_path is None or precomputed_stats_path is None:
            raise ValueError("Please provide the cache_dir or query_markers and precomputed_stats.")
        else:
            pass
    else:
        requied_files = [makers_name, stats_name]
        search_dir = Path(cache_dir) / "mapmycells" / "WMB-taxonomy" / "20240831"
        missing = [f for f in requied_files if not (search_dir/f).is_file()]
        if missing:
            abc_cache = AbcProjectCache.from_cache_dir(cache_dir)
            query_markers_path = abc_cache.get_file_path(
                directory="WMB-taxonomy",
                file_name=makers_name
                )
            
            precomputed_stats_path = abc_cache.get_file_path(
                directory="WMB-taxonomy",
                file_name=stats_name
                )
        else:
            query_markers_path = str(search_dir/makers_name)
            precomputed_stats_path = str(search_dir/stats_name)
    
    # verify the output directionary
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="map_cell_type_")
        clear_tmp = True
    else:
        clear_tmp = False

    os.makedirs(output_dir, exist_ok=True)

    # generate the tmp file
    tmp_path_h5ad = os.path.join(output_dir, f"{uuid.uuid4().hex}.h5ad")
    tmp_path_csv = os.path.join(output_dir, f"{uuid.uuid4().hex}.csv")
    tmp_path_json = os.path.join(output_dir, f"{uuid.uuid4().hex}.json")

    # pre-address the adata
    adata.X = adata.layers["counts"]
    adata.pango.save(save_path=tmp_path_h5ad)

    # the configuration for map_cell_type
    config = {
        "query_path": tmp_path_h5ad,
        "extended_result_path": tmp_path_json,
        "csv_result_path": tmp_path_csv,
        "drop_level": "CCN20230722_SUPT",
        "cloud_safe": False,
        "query_markers": {
            "serialized_lookup": query_markers_path
            },
        "precomputed_stats": {
            "path": precomputed_stats_path  
            },
        "type_assignment": {
            "normalization": "raw",
            "n_processors": 2
            }
    }

    # run the map cell type process
    runner = FromSpecifiedMarkersRunner(args=[], input_data=config)
    runner.run()

    # add the cell type to adata
    adata.obs = adata.obs.join(pd.read_csv(tmp_path_csv,
                                           comment='#')
                                           .set_index("cell_id"))

    # clear the output directionary
    if clear_tmp:
        shutil.rmtree(output_dir)

    if inplace:
        return None
    else:
        return adata

def convert_gene_symbols_to_ensembl_ids_pango(gene_symbols: list[str],
                                              gtf_path:str|None = None) -> pd.DataFrame:
    
    # verify the gtf_path
    if gtf_path is None:
        # load the EnsemblRelease for mouse
        data = EnsemblRelease("2026_04", species="mouse")
        data.download()
        data.index()  
    else:
        # load the EnsemblRelease for mouse
        data = Genome(
            reference_name="GRCm39",
            annotation_name="mouse_manual",
            annotation_version="1",
            gtf_path_or_url=gtf_path
            )
    
    # get the exact and lower case mapping of gene symbols to Ensembl IDs
    exact = defaultdict(list)
    lower = defaultdict(list)
    for gene in data.genes():
        exact[gene.name].append(gene.id)
        lower[gene.name.lower()].append(gene.id)
    
    # define the resolve function to map gene symbols to Ensembl IDs
    def resolve(name):

        ids = exact.get(name)
        if ids:
            return ids, "exact"
        
        ids = lower.get(name.lower())
        if ids:
            return ids, "casefold"
        
        return [], "missing"

    # map the gene symbols to Ensembl IDs and record the status
    mapping = {}   
    status = {}      
    for name in gene_symbols:

        ids, how = resolve(name)
        if not ids:
            status[name] = "missing"
        elif len(ids) == 1:
            mapping[name] = ids[0]
            status[name] = how
        else:
            mapping[name] = ids[0]
            status[name] = "ambiguous"

    df = pd.DataFrame({
        "gene_symbol": list(mapping.keys()),
        "ensembl_id": list(mapping.values()),
    })
        
    return df