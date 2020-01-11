import numpy as np
import h5py
import numpy
# -*- coding: utf-8 -*-

#from extract_fenlei import VGGNet
import numpy as np
import h5py
import matplotlib.image as mpimg
import argparse

import os
import os
from functools import reduce
from PIL import Image
def _binary_array_to_hex(arr):
	"""
	internal function to make a hex string out of a binary array.
	"""
	bit_string = ''.join(str(b) for b in 1 * arr.flatten())
	width = int(numpy.ceil(len(bit_string)/4))
	return '{:0>{width}x}'.format(int(bit_string, 2), width=width)
import numpy as np
import h5py
class ImageHash(object):
	"""
	Hash encapsulation. Can be used for dictionary keys and comparisons.
	"""
	def __init__(self, binary_array):
		self.hash = binary_array

	def __str__(self):
		return _binary_array_to_hex(self.hash.flatten())

	def __repr__(self):
		return repr(self.hash)

	def __sub__(self, other):
		if other is None:
			raise TypeError('Other hash must not be None.')
		if self.hash.size != other.hash.size:
			raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
		return numpy.count_nonzero(self.hash.flatten() != other.hash.flatten())


	def __eq__(self, other):
		if other is None:
			return False
		return numpy.array_equal(self.hash.flatten(), other.hash.flatten())

	def __ne__(self, other):
		if other is None:
			return False
		return not numpy.array_equal(self.hash.flatten(), other.hash.flatten())

	def __hash__(self):
		# this returns a 8 bit integer, intentionally shortening the information
		return sum([2**(i % 8) for i, v in enumerate(self.hash.flatten()) if v])


def dhash(image):
	# 将图片转化为8*8

	dhash_str = ''
	for i in range(2047):
			if image[i] > image[i+1]:
				dhash_str = dhash_str + '1'
			else:
				dhash_str = dhash_str + '0'
	result = ''
	for i in range(0, 16, 4):
		result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
	# print("dhash值",result)
	return result


# 计算两个哈希值之间的差异
def campHash(hash1, hash2):
	n = 0
	# hash长度不同返回-1,此时不能比较
	if len(hash1) != len(hash2):
		return -1
	# 如果hash长度相同遍历长度
	for i in range(len(hash1)):
		if hash1[i] != hash2[i]:
			n = n + 1
	return n

#***************************************************************
def phash_simple(image):
	"""
	Perceptual Hash computation.

	Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

	@image must be a PIL instance.
	"""
	import scipy.fftpack
	pixels = np.asarray(image)
	dct = scipy.fftpack.dct(pixels)
	dctlowfreq = dct[:]
	avg = dctlowfreq.mean()
	diff = dctlowfreq > avg
	return ImageHash(diff)
#**************************************************************
def average_hash(image):

	# find average pixel value; 'pixels' is an array of the pixel values, ranging from 0 (black) to 255 (white)
	pixels = numpy.asarray(image)
	avg = pixels.mean()

	# create string of bits
	diff = pixels > 0.5
	# make a hash
	return ImageHash(diff)

def phash(image):
	import scipy.fftpack
	pixels = numpy.asarray(image)
	dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=0)
	dctlowfreq = dct[:]
	med = numpy.median(dctlowfreq)
	diff = dctlowfreq > med
	return ImageHash(diff)

from keras.layers import Dense, GlobalAveragePooling2D
import  tensorflow as tf


def den(cb):
	tf.reset_default_graph()
	graph=tf.Graph()
	with graph.as_default() as g:
		cv = np.array([cb])
		xc = tf.convert_to_tensor(cv)
		feat1 = Dense(1024, activation='relu')(xc)
		with tf.Session(graph=g) as sess:
			sess.run(tf.global_variables_initializer())
			feat=sess.run(feat1)

	return  feat[0]

#**********************************************************



h5f = h5py.File('featureal_6_and_s.h5', 'r')
featalls = h5f['dataset_1'][:]
imgNamealls = h5f['dataset_2'][:]
h5f.close()
imgNamealls=imgNamealls.tolist()
f = open('za1.txt', 'w')
h5f = h5py.File('feature_za1.h5', 'r')
featjin = h5f['dataset_1'][:]
imgNamejin = h5f['dataset_2'][:]
h5f.close()

imgNamejin=imgNamejin.tolist()
print(len(featjin),len(imgNamejin))

if __name__ == "__main__":


	import time



	truenum = 0
	pl = 0
	#for l in range(0,1):
	for l in range(len(featjin)):
		pl=pl+1
		start = time.clock()
		print('                 ',pl)

		from sklearn.metrics.pairwise import cosine_similarity
		scores = []
		for j in range(len(featalls)):
			x=np.array(featalls[j])
			y=np.array(featjin[l])
			#***************************************************************************pi er xun 93
			x1=x-np.mean(x)
			y1=y-np.mean(y)
			scores.append((np.dot(x1,y1)/(np.linalg.norm(x1)*np.linalg.norm(y1))))
			#****************************************************************************************92

			#******************************************************************************#min shi92
			#scores.append(1-np.sqrt(np.sum(np.square(x - y))))
			#****************************************************************************************bu lei ke 90
			#up=np.sum(np.abs(np.array(featalls[j])-np.array(featjin[l])))
			#down = np.sum(np.array(featalls[j])) + np.sum(np.array(featjin[l]))
			#scores.append(1-up/down)
			#*************************************************************************************
			#scores.append(10000 - (phash_simple(featjin[l]) - phash_simple(featalls[j])))
			#scores.append((10000 - bin(int(inl) - int(featalls[i])).count('1')))
		# print(scores)
		scores = np.array(scores)

		rank_ID = np.argsort(scores)[::-1]
		rank_score = scores[rank_ID]
		# number of top retrieved images to show
		maxres = 10
		#print(imgNamejin[l].decode('utf-8').split(r"_")[0])
		for i, index in enumerate(rank_ID[0:maxres]):
			imlist =imgNamealls[index].decode('utf-8')
			all_name=imlist.split(r".")[0]
			yname=imgNamejin[l].decode('utf-8').split(r"_")[0]

			#if yname in all_name and all_name==str(yname+'_s'):
			if yname in all_name:
			   print(yname)
			   print(scores[rank_ID[i]], imlist, i)
			   f.write('\n'+yname)
			   f.write('\n'+str(i))
			   truenum=truenum+1
			   break
	print(truenum)
	f.close()



