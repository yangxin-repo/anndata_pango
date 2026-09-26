
import numpy as np

from .utils_coordinate_system_pango import (build_affine_matrix,
                                            apply_affine_matrix_to_coordinate)
class Coordinate_Pango:

    def __init__(self,
                 source_microns_per_pixel: tuple[float], # in microns (W,H)
                 source_pixel_coordinate: tuple[int,int]):     # shape(W,H)
        # build the source coordinate
        self.source_coordinate = dict()

        # pixel coordinate
        self.source_coordinate["pixel_coordinate"] = source_pixel_coordinate
        # calculate source microns per pixel
        self.source_coordinate["microns_per_pixel"] = source_microns_per_pixel

        # build the slot for affine matrix
        self.affine_matrix = dict()

        # build the coordinate mapping slot
        self.coordinate_mapping = dict()

    def shape(self):

        return self.source_coordinate["pixel_coordinate"]

    def add_affine_matrix(self,
                          affine_matrix:np.ndarray,
                          mapping_key:str):

        self.affine_matrix[mapping_key] = affine_matrix

        return None

    def crop_source_image(self,
                          crop_boundary:tuple[int,int,int,int]):

        # revise the source coordinate
        self.source_coordinate["pixel_coordinate"] = ((crop_boundary[2]-crop_boundary[0]),
                                                      (crop_boundary[3]-crop_boundary[1]))

        # build the crop affine matrix
        crop_aff_mat = build_affine_matrix(m_x=-crop_boundary[1],
                                           m_y=-crop_boundary[0])

        # revise the affine matrix and coordinate mapping
        for mp_ky in self.affine_matrix.keys():
            self.affine_matrix[mp_ky] =(np.vstack([crop_aff_mat, [0, 0, 1]]) @
                                        np.vstack([self.affine_matrix[mp_ky], [0, 0, 1]]))[:2,:]

            coordinate_new = apply_affine_matrix_to_coordinate(coordinate=self.coordinate_mapping[mp_ky],
                                                               affine_matrix=self.affine_matrix[mp_ky])
            self.coordinate_mapping[mp_ky]["pxl_row"] = coordinate_new[:,0]
            self.coordinate_mapping[mp_ky]["pxl_col"] = coordinate_new[:,1]

        return None

    def resize_source_image(self,
                            scalefactor:float):

        # calculate the new affine matrix
        for mp_ky in self.affine_matrix.keys():
            self.affine_matrix[mp_ky] = ((np.vstack([build_affine_matrix(resize=(scalefactor,scalefactor)),
                                                     [0,0,1]])) @
                                         (np.vstack([self.affine_matrix[mp_ky],
                                                     [0,0,1]])))[:2,:]

            # update the coordinate mapping
            coordinate_new = apply_affine_matrix_to_coordinate(coordinate=self.coordinate_mapping[mp_ky],
                                                               affine_matrix=self.affine_matrix[mp_ky])
            self.coordinate_mapping[mp_ky]["pxl_row"] = coordinate_new[:, 0]
            self.coordinate_mapping[mp_ky]["pxl_col"] = coordinate_new[:, 1]

        # update microns_per_pixel
        self.source_coordinate["microns_per_pixel"] = self.source_coordinate["microns_per_pixel"]/scalefactor

        # update pixel_coordinate
        self.source_coordinate["pixel_coordinate"] = (int(np.ceil((self.source_coordinate["pixel_coordinate"][0])*scalefactor)),
                                                      int(np.ceil((self.source_coordinate["pixel_coordinate"][1])*scalefactor)))

        return None
