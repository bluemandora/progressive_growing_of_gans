import h5py
import numpy as np 
import glob
from PIL import Image

clip_size = 512

def read_data():
	data = []
	files = glob.glob('datasets/healthy/*.jpg')
	ret_imgs = [] 
	names = []
	np.random.shuffle(files)
	for f in files:
		img = Image.open(f)
		img = np.array(img.resize((clip_size, clip_size), Image.ANTIALIAS))
		if img.shape != (clip_size, clip_size, 3):
			continue
		names.append(f)
		ret_imgs.append(img)
	return np.array(ret_imgs)


def convert_data(output_filename='datasets/healthy.h5'):
	output_path = output_filename
	h5file = h5py.File(output_path, mode='w')

	train_features = read_data()
	train_features = np.swapaxes(train_features, 1, 3)

	for i in range(9, -1, -1):
		shape = 2 ** i 
		h5file['data{}x{}'.format(shape, shape)] = train_features
		train_features = train_features[:, :, ::2, ::2]
		print train_features.shape


	h5file.flush()
	h5file.close()

	return (output_path,)


if __name__ == '__main__':
	convert_data()