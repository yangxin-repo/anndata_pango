
import numpy as np
import pandas as pd
import cv2

def build_affine_matrix(am_x:float = 1, # amplify x
                        am_y:float = 1, # amplify y
                        cr_x:float = 0, # crop x
                        cr_y:float = 0, # crop y
                        m_x:float = 0,  # move x
                        m_y:float = 0,  # move y
                        resize:tuple[float,float] = None): # the resize affine matrix

    # verify whether resize
    if resize is not None:
        am_x = resize[0]
        am_y = resize[1]
        m_x = (am_x-1)/2
        m_y = (am_y-1)/2

    # build the affine matrix
    affine_matrix = np.array([[am_x,cr_x,m_x],
                              [cr_y,am_y,m_y]],
                             dtype=np.float64)

    return affine_matrix

def apply_affine_matrix_to_coordinate(coordinate:pd.DataFrame,
                                      affine_matrix:np.ndarray):

    # extract the coordinate
    points = coordinate[["array_row", "array_col"]].to_numpy(dtype=np.float64)
    ones = np.ones((len(points), 1))
    pts = np.hstack([points, ones])

    # build the affine matrix
    af_mat = np.vstack([affine_matrix, [0, 0, 1]])

    # calculate the mapping coordinate
    affined_pts = pts @ af_mat.T

    return affined_pts

def estimate_affine_matrix_from_coordinate_mapping(coordinate_mapping:pd.DataFrame):

    # extract the source points
    src_pts = np.float32(coordinate_mapping[["array_row","array_col"]].to_numpy())

    # extract the destination points
    dst_pts = np.float32(coordinate_mapping[["pxl_row","pxl_col"]].to_numpy())

    M, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    return M