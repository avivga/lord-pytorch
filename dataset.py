import os
import re
from abc import ABC, abstractmethod

import numpy as np
import imageio
import cv2
import dlib
import h5py


supported_datasets = [
	'smallnorb',
	'cars3d',
	'shapes3d',
	'celeba',
	'kth',
	'rafd'
]


def get_dataset(dataset_id, path=None):
	if dataset_id == 'smallnorb':
		return SmallNorb(path)

	if dataset_id == 'cars3d':
		return Cars3D(path)

	if dataset_id == 'shapes3d':
		return Shapes3D(path)

	if dataset_id == 'celeba':
		return CelebA(path)

	if dataset_id == 'kth':
		return KTH(path)

	if dataset_id == 'rafd':
		return RaFD(path)

	raise Exception('unsupported dataset: %s' % dataset_id)


class DataSet(ABC):

	def __init__(self, base_dir=None):
		super().__init__()
		self._base_dir = base_dir

	@abstractmethod
	def read_images(self):
		pass


class SmallNorb(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def __list_imgs(self):
		img_paths = []
		class_ids = []
		content_ids = []

		regex = re.compile('azimuth(\d+)_elevation(\d+)_lighting(\d+)_(\w+).jpg')
		for category in os.listdir(self._base_dir):
			for instance in os.listdir(os.path.join(self._base_dir, category)):
				for file_name in os.listdir(os.path.join(self._base_dir, category, instance)):
					img_path = os.path.join(self._base_dir, category, instance, file_name)
					azimuth, elevation, lighting, lt_rt = regex.match(file_name).groups()

					class_id = '_'.join((category, instance, elevation, lighting, lt_rt))
					content_id = azimuth

					img_paths.append(img_path)
					class_ids.append(class_id)
					content_ids.append(content_id)

		return img_paths, class_ids, content_ids

	def read_images(self):
		img_paths, class_ids, content_ids = self.__list_imgs()

		unique_class_ids = list(set(class_ids))
		unique_content_ids = list(set(content_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 1), dtype=np.uint8)
		classes = np.empty(shape=(len(img_paths), ), dtype=np.uint32)
		contents = np.empty(shape=(len(img_paths), ), dtype=np.uint32)

		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])
			imgs[i, :, :, 0] = cv2.resize(img, dsize=(64, 64))

			classes[i] = unique_class_ids.index(class_ids[i])
			contents[i] = unique_content_ids.index(content_ids[i])

		return imgs, classes, contents


class Cars3D(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__data_path = os.path.join(base_dir, 'cars3d.npz')

	def read_images(self):
		imgs = np.load(self.__data_path)['imgs']
		classes = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)
		contents = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)

		for elevation in range(4):
			for azimuth in range(24):
				for object_id in range(183):
					img_idx = elevation * 24 * 183 + azimuth * 183 + object_id

					classes[img_idx] = object_id
					contents[img_idx] = elevation * 24 + azimuth

		return imgs, classes, contents


class Shapes3D(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__data_path = os.path.join(base_dir, '3dshapes.h5')

	def __img_index(self, floor_hue, wall_hue, object_hue, scale, shape, orientation):
		return (
			floor_hue * 10 * 10 * 8 * 4 * 15
			+ wall_hue * 10 * 8 * 4 * 15
			+ object_hue * 8 * 4 * 15
			+ scale * 4 * 15
			+ shape * 15
			+ orientation
		)

	def read_images(self):
		with h5py.File(self.__data_path, 'r') as data:
			imgs = data['images'][:]
			classes = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)
			content_ids = dict()

			for floor_hue in range(10):
				for wall_hue in range(10):
					for object_hue in range(10):
						for scale in range(8):
							for shape in range(4):
								for orientation in range(15):
									img_idx = self.__img_index(floor_hue, wall_hue, object_hue, scale, shape, orientation)
									content_id = '_'.join((str(floor_hue), str(wall_hue), str(object_hue), str(scale), str(orientation)))

									classes[img_idx] = shape
									content_ids[img_idx] = content_id

			unique_content_ids = list(set(content_ids.values()))
			contents = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)
			for img_idx, content_id in content_ids.items():
				contents[img_idx] = unique_content_ids.index(content_id)

			return imgs, classes, contents


class CelebA(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__imgs_dir = os.path.join(self._base_dir, 'Img', 'img_align_celeba_png.7z', 'img_align_celeba_png')
		self.__identity_map_path = os.path.join(self._base_dir, 'Anno', 'identity_CelebA.txt')

	def __list_imgs(self):
		with open(self.__identity_map_path, 'r') as fd:
			lines = fd.read().splitlines()

		img_paths = []
		class_ids = []

		for line in lines:
			img_name, class_id = line.split(' ')
			img_path = os.path.join(self.__imgs_dir, os.path.splitext(img_name)[0] + '.png')

			img_paths.append(img_path)
			class_ids.append(class_id)

		return img_paths, class_ids

	def read_images(self, crop_size=(128, 128), target_size=(64, 64)):
		img_paths, class_ids = self.__list_imgs()

		unique_class_ids = list(set(class_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 3), dtype=np.uint8)
		classes = np.empty(shape=(len(img_paths), ), dtype=np.uint32)
		contents = np.zeros(shape=(len(img_paths), ), dtype=np.uint32)

		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])

			if crop_size:
				img = img[
					(img.shape[0] // 2 - crop_size[0] // 2):(img.shape[0] // 2 + crop_size[0] // 2),
					(img.shape[1] // 2 - crop_size[1] // 2):(img.shape[1] // 2 + crop_size[1] // 2)
				]

			if target_size:
				img = cv2.resize(img, dsize=target_size)

			imgs[i] = img
			classes[i] = unique_class_ids.index(class_ids[i])

		return imgs, classes, contents


class KTH(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__action_dir = os.path.join(self._base_dir, 'handwaving')
		self.__condition = 'd4'

	def __list_imgs(self):
		img_paths = []
		class_ids = []

		for class_id in os.listdir(self.__action_dir):
			for f in os.listdir(os.path.join(self.__action_dir, class_id, self.__condition)):
				img_paths.append(os.path.join(self.__action_dir, class_id, self.__condition, f))
				class_ids.append(class_id)

		return img_paths, class_ids

	def read_images(self):
		img_paths, class_ids = self.__list_imgs()

		unique_class_ids = list(set(class_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 1), dtype=np.uint8)
		classes = np.empty(shape=(len(img_paths), ), dtype=np.uint32)
		contents = np.zeros(shape=(len(img_paths), ), dtype=np.uint32)

		for i in range(len(img_paths)):
			imgs[i, :, :, 0] = cv2.cvtColor(cv2.imread(img_paths[i]), cv2.COLOR_BGR2GRAY)
			classes[i] = unique_class_ids.index(class_ids[i])

		return imgs, classes, contents


class RaFD(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def __list_imgs(self):
		img_paths = []
		expression_ids = []

		regex = re.compile('Rafd(\d+)_(\d+)_(\w+)_(\w+)_(\w+)_(\w+).jpg')
		for file_name in os.listdir(self._base_dir):
			img_path = os.path.join(self._base_dir, file_name)
			idx, identity_id, description, gender, expression_id, angle = regex.match(file_name).groups()

			img_paths.append(img_path)
			expression_ids.append(expression_id)

		return img_paths, expression_ids

	def read_images(self):
		img_paths, expression_ids = self.__list_imgs()

		unique_expression_ids = list(set(expression_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 3), dtype=np.uint8)
		expressions = np.empty(shape=(len(img_paths), ), dtype=np.uint32)

		face_detector = dlib.get_frontal_face_detector()
		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])

			detections, scores, weight_indices = face_detector.run(img, upsample_num_times=0, adjust_threshold=-1)
			face_bb = detections[np.argmax(scores)]

			top = max((face_bb.bottom() + face_bb.top()) // 2 - 681 // 2, 0)
			face = img[top:(top + 681), :]

			imgs[i] = cv2.resize(face, dsize=(64, 64))
			expressions[i] = unique_expression_ids.index(expression_ids[i])

		return imgs, expressions, np.zeros_like(expressions)
