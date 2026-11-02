
from .image_pango.image_pango import Image_Pango
from .anndata_pango import load_10x_h5_pango
from .anndata_pango import concat_pango
from .anndata_pango import get_the_rank_genes_pango
from .anndata_pango import map_cell_type_pango

# register Pango_Accessor
from .anndata_pango import Pango_Accessor

def main() -> None:
    print("Hello from anndata_pango!")
