import os
import tarfile
import h5py
import numpy
import six
from six.moves import range, cPickle
from scipy.misc import imresize

DISTRIBUTION_FILE = 'cifar-10-python.tar.gz'

def convert_cifar10(output_filename='cifar10.h5'):
	"""Converts the CIFAR-10 dataset to HDF5.
	Converts the CIFAR-10 dataset to an HDF5 dataset compatible with
	:class:`fuel.datasets.CIFAR10`. The converted dataset is saved as
	'cifar10.hdf5'.
	It assumes the existence of the following file:
	* `cifar-10-python.tar.gz`
	Parameters
	----------
	directory : str
		Directory in which input files reside.
	output_directory : str
		Directory in which to save the converted dataset.
	output_filename : str, optional
		Name of the saved dataset. Defaults to 'cifar10.hdf5'.
	Returns
	-------
	output_paths : tuple of str
		Single-element tuple containing the path to the converted dataset.
	"""
	output_path = output_filename
	h5file = h5py.File(output_path, mode='w')
	input_file =  DISTRIBUTION_FILE
	tar_file = tarfile.open(input_file, 'r:gz')

	train_batches = []
	for batch in range(1, 6):
		file = tar_file.extractfile(
			'cifar-10-batches-py/data_batch_%d' % batch)
		try:
			if six.PY3:
				array = cPickle.load(file, encoding='latin1')
			else:
				array = cPickle.load(file)
			train_batches.append(array)
		finally:
			file.close()

	train_features = numpy.concatenate(
		[batch['data'].reshape(batch['data'].shape[0], 3, 32, 32)
			for batch in train_batches])
	train_labels = numpy.concatenate(
		[numpy.array(batch['labels'], dtype=numpy.uint8)
			for batch in train_batches])
	train_labels = numpy.expand_dims(train_labels, 1)

	for i in [32, 16, 8, 4, 2, 1]:
		h5file['data{}x{}'.format(i, i)] = train_features
		train_features = train_features[:, :, ::2, ::2]
		print train_features.shape

	numpy.save('cifar10-32-labels.npy', train_labels)


	h5file.flush()
	h5file.close()

	return (output_path,)


if __name__ == '__main__':
	convert_cifar10()