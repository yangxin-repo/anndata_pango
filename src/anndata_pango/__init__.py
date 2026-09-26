
from .image_pango.image_pango import Image_Pango
from .anndata_pango import load_10x_h5_pango

# register Pango_Accessor
from .anndata_pango import Pango_Accessor

def main() -> None:
    print("Hello from anndata_pango!")
