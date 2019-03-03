import os


class CelebA:

	def __init__(self, base_dir):
		self.__base_dir = base_dir
		self.__imgs_dir = os.path.join(self.__base_dir, 'Img', 'img_align_celeba_png.7z', 'img_align_celeba_png')
		self.__identity_map_path = os.path.join(self.__base_dir, 'Anno', 'identity_CelebA.txt')

	def get_identity_map(self):
		with open(self.__identity_map_path, 'r') as fd:
			lines = fd.read().splitlines()

		identity_map = dict()
		for line in lines:
			img_name, identity = line.split(' ')
			img_path = os.path.join(self.__imgs_dir, os.path.splitext(img_name)[0] + '.png')

			if identity not in identity_map:
				identity_map[identity] = list()

			identity_map[identity].append(img_path)

		return identity_map
