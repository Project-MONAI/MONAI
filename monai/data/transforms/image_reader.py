import logging
import numpy as np


class ImageReader(object):
    """Base class for Image Loader."""

    def __init__(self, dtype=np.float32):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._dtype = dtype

    def _read_from_file_list(self, file_names):
        raise NotImplementedError('{} cannot load from file list'.format(self.__class__.__name__))

    def _read_from_file(self, file_name):
        raise NotImplementedError('{} cannot load from file'.format(self.__class__.__name__))

    def read(self, file_name_spec):
        if isinstance(file_name_spec, np.ndarray):
            file_name_spec = file_name_spec.tolist()
        if isinstance(file_name_spec, list):
            assert len(file_name_spec) > 0, 'file_name_spec must not be empty list'

            file_names = []
            for file_name in file_name_spec:
                if isinstance(file_name, (bytes, bytearray)):
                    file_name = file_name.decode('UTF-8')
                file_names.append(file_name)

            result = self._read_from_file_list(file_names)
        else:
            file_name = file_name_spec
            if isinstance(file_name, (bytes, bytearray)):
                file_name = file_name.decode('UTF-8')
            assert isinstance(file_name, str), 'file_name_spec must be a str'
            assert len(file_name) > 0, 'file_name_spec must not be empty'
            result = self._read_from_file(file_name)

        return result
