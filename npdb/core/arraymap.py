"""
Implements a memory map of a dbarray on disk.
"""

import os
import math

import numpy as np

class arraymap(object):
    """
    Maps dbarray elements to locations on disk.
    """
    def __init__(self, shape, dtype, byteorder, row_major, data_dir,
                 max_file_size):
        """
        Creates file system for a dbarray, and establishes mapping from disk 
        locations to array indices. 
        """
        # size and shape of data
        self.shape = tuple(shape)
        self.size = reduce(operator.mul, self.shape)
        self.ndim = len(shape)

        # data elements
        self.dtype = np.dtype(dtype)
        self.itemsize = self.dtype.itemsize

        # data layout
        self.byteorder = byteorder
        self.row_major = row_major

        # maximum file size
        self.max_file_size = max_file_size

        # create directory and file structure
        self.file_paths, self.bytes_per_file, self.array_dir_name = setup_dirs(
            data_dir, self.size, self.itemsize, self.max_file_size)
        self.n_files = len(self.file_paths)

        # cache potentially repeatedly used calculations
        self.partial_prod_shape = None

    def __repr__(self):
        return "arraymap " + self.array_dir_name

    def setup_dirs(data_dir, size, itemsize, max_file_size):
        """
        Creates directory to contain array data and files to contain array 
        buffers
        """
        assert(max_file_size > 1), "files must be at least one byte large"

        # disk storage locations
        if data_dir is None:
            # put data root file in working directory
            data_dir_cwd = os.path.join(os.getcwd(), "data")
            if not os.path.isdir(data_dir_cwd):
                os.makedirs(data_dir_cwd)
            data_dir = data_dir_cwd

        array_dir_name_len = 12
        letters = string.ascii_lowercase
        array_dir_name =  (''.join(random.choice(letters) for i in range(
                           array_dir_name_len)))
        array_dir = os.path.join(data_dir, array_dir_name)
        while os.path.isdir(array_dir):
            array_dir_name =  (''.join(random.choice(letters) for i in range(
                               array_dir_name_len)))
            array_dir = os.path.join(data_dir, array_dir_name)

        total_bytes = size * itemsize
        n_files = math.ceil(total_bytes / float(max_file_size))
        even_bpf = total_bytes / float(n_files)

        if even_bpf < 1:
            # sub-one byte; special case to avoid creating two files
            bytes_per_file = {0: even_bpf}
        else:
            # regular case
            floor_even_bpf = math.floor(even_bpf)
            bytes_per_file = {filenum: floor_even_bpf for filenum in 
                              range(n_files - 1)}
            # last file rounds out total_bytes
            bytes_per_file[n_files - 1] = total_bytes - sum(
                                          bytes_per_file.values())

        file_ext = ".dat"
        file_paths = {filenum: os.path.join(array_dir, ("buffer_" + 
                      str(filenum) + "." + file_ext)) for filenum in range(
                      n_files)}

        return file_paths, bytes_per_file, array_dir_name

    def index_to_address(self, index):
        """
        Maps a simple single-element dbarray index to a disk address.

        Args:
            index (tuple): tuple of scalar locations along each axis of a 
            dbarray

        Returns:
            address (tuple): the corresponding file address, expressed as a 
            (filenum, offset)
        """
        # calculate partial product of dbarray array shape
        if self.partial_prod_shape is None:
            axis_mults = [1] * self.ndim
            for axis in range(self.ndim - 1):
                axis_mults[axis + 1] = axis_mults[axis] * self.shape[axis]
            self.partial_prod_shape = reversed(axis_mults)
        
        # calculate item number in C-contiguous order
        item_num = np.dot(np.array(axis_mults), np.array(index))
        total_byte_offset = self.itemsize * item_num

        # determine file number and offset within file
        if self.n_files == 1:
            filenum, offset = 0, total_byte_offset
        else:
            bpf = self.bytes_per_file[0]
            filenum, offset = total_byte_offset / bpf, total_byte_offset % bpf
            if filenum == 0:
                # zero-index correction
                offset -= 1

        return filnum, offset

    def get_addresses(self, index_bounds):
        """
        Returns the file addresses of the data corresponding to 
        index_bounds.

        A single address takes the form of a tuple of (filenum, offset) tuples 
        such that a maximal C-contiguous segment of selected data exists between
        the address elements of the tuple.

        Args:
            index_bounds (list): a list of bounding index tuples of each maximal
            C-contiguous block of data to be selected.

        Returns:
            address_bounds (list): a list of starting and ending address tuples.
        """
        address_bounds = [(index_to_address(start), index_to_address(stop)) for
                          start, stop in index_bounds]

        return address_bounds

    def pull(self, index_bounds, shape):
        """
        Returns ndarray corresponding to the specified bounds.
        """
        # contiguous blocks described by (filenum, offset) ranges
        flattened_locs = get_addresses(index_bounds)

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
                fp = open(os.path.join(self.root_dir, self.file_paths[filenum]), "rb")
                last_fp = fp

            block_length = end_offset - start_offset
            unraveled_start = unraveled_end
            unraveled_end = unraveled_start + block_length

            # read between starting offset and ending offset
            fp.seek(start_offset)
            vals = fp.read(block_length)
            np.put(pulled, range(unraveled_start, unraveled_end), vals)
            
        return pulled

    def push(self, dbview):
        """
        Writes ndarray to disk at location specified by bounds.
        """
        ndarray = dbview.asndarray()
        arrslice, index_bounds

        flattened_locs = idxs_to_locs(index_bounds)

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
                fp = open(os.path.join(self.root_dir, self.file_paths[filenum]), "wb")
                last_fp = fp


            fp.seek(start_offset)





print "here"
index_bounds = [((0,0),(0,1)),((1,1),(1,2)),((1,2),(3,4)),((0,1),(0,0))]
print maximize(index_bounds)


index_bounds = [((0,0),(0,0)),((1,1),(1,2)),((2,2),(3,4)),((0,1),(0,0))]
print maximize(index_bounds)


index_bounds = [((0,0),(0,1)),((1,1),(1,2)),((2,2),(0,1)),((0,1),(0,0))]
print maximize(index_bounds)


index_bounds = [((0,0),(0,1)),((0,1),(1,2)),((1,2),(0,1)),((0,1),(0,0))]
print maximize(index_bounds)


index_bounds = [((0,0),(0,0)),((0,1),(1,2)),((1,2),(0,1)),((0,1),(0,0))]
print maximize(index_bounds)


index_bounds = [((0,0),(1,1)),((1,1),(2,2)),((2,2),(3,3)),((3,3),(3,3))]
print maximize(index_bounds)


