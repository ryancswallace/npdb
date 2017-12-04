import os

import numpy as np

class arraymap(object):
    """
    Maps dbarray elements to locations on disk.
    """

    def __init__(self, shape, dtype, data_dir, max_chunk_size):
        """
        Creates file system for a dbarray, and establishes mapping from disk locations to 
        the array. 
        """
        # root_dir = 
        # file_ext = ".dat"
        # dsize = # bytes per dtype value
        pass

    def idx_to_loc(self, start_idx, end_idx):
        """
        Returns list of (filenum, offset) tuples within the file corresponding to thhe data
        between start_idx and end_idx
        """
        filenum = 0
        offset = 0

        return [filenum, offset]

    def pull(self, flattened_bounds, shape):
        """
        Returns ndarray corresponding to the specified bounds.
        """
        flattened_locs_nested = [self.idx_to_loc(start_idx, end_idx) for start_idx, end_idx in flattened_bounds]
        flattened_locs = [loc for nested in flattened_locs_nested for loc in nested]

        pulled = np.empty(shape, dtype=self.dtype)

        unraveled_start, unraveled_end = 0, 0
        last_filenum, last_fp = None, None
        for (start_filenum, start_offset), (end_filenum, end_offset) in flattened_locs:
            filenum = start_filenum
            if last_filenum == filenum:
                # file is already open
                fp = last_fp
            else:
                # close previous file
                last_fp.close()
                # open new file
                fp = open(os.path.join(self.root_dir, filenum + self.file_ext), "rb")
                last_fp = fp

            block_length = end_offset - start_offset
            unraveled_start = unraveled_end
            unraveled_end = unraveled_start + block_length

            # read between starting offset and ending offset
            fp.seek(start_offset)
            vals = fp.read(block_length)
            np.put(pulled, range(unraveled_start, unraveled_end), vals)
            
        return pulled

    def push(self, ndarray, flattened_bounds):
        """
        Writes ndarray to disk at location specified by bounds.
        """
        pass
