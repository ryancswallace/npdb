"""
__init__ file for npdb package.

Three functions:
	1) Explains omission of __all__ variable.
	2) Configures user parameters.
	3) Imports user API into npdb namespace.
"""
###############
### __all__ ###
###############
# The __all__ variable is not defined for the npdb package, as a large number
# of the npdb names would conflict with those in the numpy namespace. To avoid
# potential conflicts, npdb should be imported as "import npdb", or if brevity
# is important, as "import npdb as nd"

###############################
### Parameter configuration ### 
###############################
# TODO: work this out
# from npdb.core import config 

# def get_def_max_file_size():
# 	return config.MAX_FILE_SIZE

# def get_def_max_view_size():
# 	return config.MAX_VIEW_SIZE

################
### User API ###
################
### core modules and classes
from npdb import core
from npdb import creation
from npdb import indexing
from npdb import linalg
from npdb import math
from npdb import random
from npdb import sorting
from npdb import statistics

from npdb.core.core import dbarray
from npdb.core.core import dbview

from npdb.indexing import indexing

#### array creation routines
# from npdb.creation.basic import empty
# from npdb.creation.basic import empty_like
# from npdb.creation.basic import eye
# from npdb.creation.basic import full
# from npdb.creation.basic import full_like
# from npdb.creation.basic import identity
# from npdb.creation.basic import ones
# from npdb.creation.basic import ones_like
# from npdb.creation.basic import zeros
# from npdb.creation.basic import zeros_like

# from npdb.creation.from_data import array
# from npdb.creation.from_data import asanyarray
# from npdb.creation.from_data import asarray

# from npdb.creation.ranges import arange
# from npdb.creation.ranges import linspace
# from npdb.creation.ranges import logspace
# from npdb.creation.ranges import meshgrid
# from npdb.creation.ranges import mgrid
# from npdb.creation.ranges import ogrid

# from npdb.creation.matrix import diag
# from npdb.creation.matrix import diagflat

### array manipulation
# from npdb.manipulation.basic import copyto

# from npdb.manipulation.shape import ravel
# from npdb.manipulation.shape import reshape

# from npdb.manipulation.transpose import moveaxis
# from npdb.manipulation.transpose import rollaxis
# from npdb.manipulation.transpose import swapaxes
# from npdb.manipulation.transpose import transpose

# from npdb.manipulation.dims import atleast_1d
# from npdb.manipulation.dims import atleast_2d
# from npdb.manipulation.dims import atleast_3d
# from npdb.manipulation.dims import broadcast
# from npdb.manipulation.dims import broadcast_arrays
# from npdb.manipulation.dims import broadcast_to
# from npdb.manipulation.dims import expand_dims
# from npdb.manipulation.dims import squeeze

# from npdb.manipulation.join import column_stack
# from npdb.manipulation.join import concatenate
# from npdb.manipulation.join import dstack
# from npdb.manipulation.join import hstack
# from npdb.manipulation.join import stack
# from npdb.manipulation.join import vstack

# from npdb.manipulation.kind import asfortranarray

# from npdb.manipulation.split import array_split
# from npdb.manipulation.split import dsplit
# from npdb.manipulation.split import hsplit
# from npdb.manipulation.split import split
# from npdb.manipulation.split import vsplit

# from npdb.manipulation.tiling import repeat
# from npdb.manipulation.tiling import tile

# from npdb.manipulation.rearrange import flip
# from npdb.manipulation.rearrange import fliplr
# from npdb.manipulation.rearrange import flipud
# from npdb.manipulation.rearrange import roll
# from npdb.manipulation.rearrange import rot90

### indexing
