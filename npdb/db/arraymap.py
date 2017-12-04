import os

class arraymap(object):
    """
    Maps dbarray elements to locations on disk.
    """

    def __init__(self, shape, dtype, filename, max_chunk_size):
        """
        Creates file system for a dbarray, and establishes mapping from disk locations to 
        the array. 
        """
        # filepath = D # pointer to file

        # return filepath
        pass

    def pull(self, flattened_bounds):
        """
        Returns ndarray corresponding to the specified bounds.
        """
        pass

    def push(self, ndarray, flattened_bounds):
        """
        Writes ndarray to disk at location specified by bounds.
        """
        pass
