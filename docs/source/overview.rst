.. _overview:

Overview
========

.. module:: npdb

npdb is an implementation of large disk-stored NumPy-compatible n-dimenstional arrays that may exceed available memory. npdb implements the core multi-dimensional array class :class:`npdb.dbarray`, which supports persistent binary storage and distributed batch processed operations. :class:`npdb.dbarray` supports a subset of the :class:`numpy.ndarray` interface.

Currently, npdb supports the following subset of the NumPy interface:

- Basic indexing

npdb extends NumPy's capabilities with the folloiwng:

- Support for disk-stored arrays that exceed the size of memory
- Native parallelization of most methods for both speed and memory constraints
