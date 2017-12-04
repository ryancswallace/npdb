"""
Implements indexing on a dbarray.
"""

class dbindex(object):
    """
    A dbindex is a one of a slice, an int, or a tuple of slices and ints.
    """
    def __init__(self, slc):
        """
        Stores slice and loosely checks that the type is workable.
        """        
        # try to cast slc as a workable type
        if not type(slc) in [slice, int, tuple]:
            try:
                slc = slice(slc)
            except TypeError as e:
                print "dbindex must be one of a slice, an int, or a tuple of slices and ints", e
                sys.exit(e)

        self.arrslice = slc
        self.slicetype = type(self.arrslice)

        # cache transformation operations
        self.flattened_bounds = None
        self.bounding_hypercube = None
        self.shape = None
        self.offset = None

    def __repr__(self):
        pass

    @staticmethod
    def flattened_bounds(self, arrslice, db_shape):
        """
        Returns a list of one tuple for each contiguous section of the dbarray described by the dbindex.
        Each tuple contains the index in the dbarray of the first and last elements of the contiguous secion.
        Sections appear in the order they occur in the slice. 

        Args:
            arrslice (dbindex): a slice of a dbarray of shape db_shape
            db_shape (dbarray): shape of a dbarray

        Returns:
            flattened_bounds (list): a list of tuples for each contiguous block of data in arrslice (C-ordered). 
            Each tuple (start, stop) indicates the index of the first and last elements of the contigous block.
        """
        if not arrslice.flattened_bounds is None:
            # cached
            return arrslice.flattened_bounds

        else:
            # needs to compute 
            # number of dimensions in view of dbarray
            ndim = len(db_sape)

            flattened_bounds = []
            if arrslice.slicetype is int:
                # index by a scalar
                if ndim == 1:
                    # slice is single value
                    return [(arrslice, arrslice)]
                elif ndim == 2:
                    # slice is 1D array
                    return [(0, db_shape[1])]
                else:
                    # slice is n-dim array, n > 1
                    sub_shape = db_shape[1:]
                    sub_bounds = []
                    for d in db_shape[1]:
                        new_slice = dbindex(slice(None, None, None))
                        sub_bounds += flattened_bounds(new_slice, subshape)
                    return [(arrslice + start, arrslice + end) for start, end in sub_bounds]

            elif arrslice.slicetype is slice:
                # must parse slice values to scalars

            else:
                # dbindex is a tuple


            arrslice.flattened_bounds = flattened_bounds

            return flattened_bounds

    @staticmethod
    def bounding_hypercube(self, arrslice):
        """
        Calculates indices of minimum hypercube bounding the dbindex arrslice.
        """
        if not self.bounding_hypercube is None:
            # cached
            return self.bounding_hypercube

        else:
            # needs to compute
            bounding_hypercube = 1
            self.bounding_hypercube = bounding_hypercube

            return bounding_hypercube

    @staticmethod
    def shape(self, arrslice):
        """
        Returns dimensions of minimum bounding_hypercube of arrslice.
        """


    @staticmethod
    def offset(self, arrslice):
        """
        Returns index in dbarray object along all dimensions of the [0,...,0] element of the slice.
        """
        if not self.flattened_bounds is None:
            # cached
            return self.flattened_bounds

        else:
            # needs to compute
            offset = 1
            self.offset = offset

            return offset

    @staticmethod
    def dbindex_from_bounds(self, bounds):
        """
        Not sure if we need this.

        Returns dbindex corresponding to linear of hypercube bounds.
        """
        pass