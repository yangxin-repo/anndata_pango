
import anndata as ad

from .anndata_pango import concat_pango, get_the_rank_genes_pango


def scvi_pipeline_stage1(*,
                          h5ad_path_list:list[str],
                          batch_symbol:list[str],
                          save_directory:str,
                          batch_key:str,
                          categorical_covariate_keys:list[str]|None=None,
                          continuous_covariate_keys:list[str] | None=None) -> ad.AnnData:
    
    # load the AnnData object from the h5ad file
    adata_ls = [ad.read_h5ad(path) for path in h5ad_path_list]
    
    # extract the characteristic gene set
    gene_set = set()
    for ad_obj in adata_ls:
        ge_ls = get_the_rank_genes_pango(ad_obj,gene_num=300)
        gene_set.update(ge_ls)
    gene_set = list(gene_set)
    
    adata_ls = [ad_obj[:,ad_obj.var_names.isin(gene_set)] for ad_obj in adata_ls]
    
    # concat the adata list
    adata = concat_pango(adata_list=adata_ls,
                         batch_key=batch_symbol,
                         normalize=False,
                         highly_variable_genes=False)
    # fit the scVI model to the AnnData object
    model_scvi = adata.pango.fit_scvi_models(batch_key=batch_key,
                                             highly_variable_genes=False,
                                             categorical_covariate_keys=categorical_covariate_keys,
                                             continuous_covariate_keys=continuous_covariate_keys)
    
    # build the graph-related space in the AnnData object
    adata.pango.build_graph_related_space()
    
    # save the scVI model and AnnData object to the specified directory
    model_scvi.save(f"{save_directory}/scvi_model.pt", 
                    overwrite=True)
    
    adata.pango.save(f"{save_directory}/scvi_adata.h5ad")
    
    # save the training and validation results to a parquet file
    train_test_results = model_scvi.history['elbo_train']
    train_test_results['elbo_validation'] = model_scvi.history['elbo_validation']
    train_test_results.to_parquet(f"{save_directory}/scvi_train_test_results.parquet", 
                                  index=False)

    return adata

def scvi_pipeline_stage2():
    
    return