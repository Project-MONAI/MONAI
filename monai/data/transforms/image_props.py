class ImageProperty:
    """Key names for image properties.

    """
    DATA = 'data'
    FILENAME = 'file_name'
    AFFINE = 'affine'  # image affine matrix
    ORIGINAL_SHAPE = 'original_shape'
    ORIGINAL_SHAPE_FORMAT = 'original_shape_format'
    SPACING = 'spacing'  # itk naming convention for pixel/voxel size
    FORMAT = 'file_format'
    NIFTI_FORMAT = 'nii'
    IS_CANONICAL = 'is_canonical'
    SHAPE_FORMAT = 'shape_format'
    BACKGROUND_INDEX = 'background_index'  # which index is background
