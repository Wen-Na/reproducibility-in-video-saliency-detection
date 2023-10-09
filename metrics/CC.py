import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import cv2
import re


# My way --> You can use the function corrcoef in numpy to find Peason correlation. First you need to flatten both image arrays
def main():

	technique = '_dsl' # DA or Base Model (HD)?
	perf_metric = 'CC_results'
	separated_directories = True # are saliency maps in separated directories? --> there are cases where saliency directory contains two subdirectories: frames (sal. prediction) and images (sal. prediction overlayed on the original video frame)

	# set paths to saliency maps and fixation maps
	path_saliency_maps = '../predicted_sal_maps_DIEM17videos_dataset_grey_scale'
	path_fixation_maps = '/nfs/home/navarretee/miniconda3/envs/vinet_env/vinet/DIEM_testset_17videos'
	path_results = 'output_ev_metrics/DIEM17videos_dataset_weigths3rd' 	# set path where to store the output results
	if not os.path.isdir(path_results):
		os.makedirs(path_results)
	path_extra_info = path_results + '/extra_info_CC_results.txt'

	# retrieve directories with the several folders that contain the video frames
	list_saliency_maps = [d for d in os.listdir(path_saliency_maps) if os.path.isdir(os.path.join(path_saliency_maps, d))]
	list_saliency_maps.sort()
	list_fixation_maps = [d for d in os.listdir(path_fixation_maps) if os.path.isdir(os.path.join(path_fixation_maps, d))]
	list_fixation_maps.sort()

	# loop through each folder and save results as a list
	results_list = []
	for dname in list_fixation_maps:

		video_name = dname

		if separated_directories:
			# path to video frames
			path_predicted = os.path.join(path_saliency_maps, video_name, 'frames')  # prediction/saliency maps
			#path_gt = os.path.join(path_fixation_maps, dname)  # ground truth/fixations maps
			path_gt = os.path.join(path_fixation_maps, dname,'maps')  # ground truth/fixations maps
		else:
			# path to video frames
			path_predicted = os.path.join(path_saliency_maps, video_name)  # prediction/saliency maps
			path_gt = os.path.join(path_fixation_maps, dname)  # ground truth/fixations maps

		# list of frames ground truth
		frames_gt_list = [f for f in listdir(path_gt) if isfile(join(path_gt, f))]
		frames_gt_list = [val for val in frames_gt_list if (val.endswith(".png") or val.endswith(".jpg"))]
		frames_gt_list.sort()  # sort

		# list of frames predicted
		frames_predicted_list = [f for f in listdir(path_predicted) if isfile(join(path_predicted, f))]
		frames_predicted_list = [val for val in frames_predicted_list if (val.endswith(".png") or val.endswith(".jpg"))]
		frames_predicted_list.sort()  # sort

		# just for registration: detect when ground truth frames are less than predicted frames (because there was not enough data to create the gt/fixation maps)
		if len(frames_gt_list) < len(frames_predicted_list):
			# save information
			with open(path_extra_info, 'a') as file:
				print('video:', dname, 'number of gt/fixation maps less than predicted frames. ~Processed frames:', len(frames_gt_list), file=file)
		elif len(frames_gt_list) == len(frames_predicted_list):
			with open(path_extra_info, 'a') as file:
				print('video:', dname, 'number of gt/fixation maps same as predicted frames . ~Processed frames:', len(frames_gt_list), file=file)
		else: # if	len(frames_gt_list) > len(frames_predicted_list)
			with open(path_extra_info, 'a') as file:
				print('video:', dname, 'number of gt/fixation maps more than predicted frames. ~Processed frames:', len(frames_predicted_list), file=file)

		# compute metric
		metric_list = []
		for i in range(len(frames_gt_list)):

			# sometimes # of ground truth (gt/fixation maps) are more than saliency/fixation maps (because participants were longer in the computer), so, detect this and finish the loop
			#if i + 1 > len(frames_predicted_list):
			#	break

			# make sure that that background frame has corresponding salience/fixation map (sometimes there is no corresponding
			# fixation map because there was not enough data to create the fixation/density plot. Compare exluding the extensions, e.g., 'jpg and png
			#frame_number = str(int(frames_gt_list[i].split('.')[0]))
			frame_number_reg_exp_gt = re.match(r'.*?(\d+).*', frames_gt_list[i])  # regular expression
			frame_number_gt = str(int(frame_number_reg_exp_gt.group(1)))  # extract reg exp and make it integer
			#if any(frame_number == str(int(file.split('.')[0])) for file in frames_predicted_list):
			if any(frame_number_gt == str(int(re.match(r'.*?(\d+).*', file).group(1))) for file in frames_predicted_list):
				# upload images
				image_gt = cv2.imread(path_gt + '/' + frames_gt_list[i], cv2.IMREAD_GRAYSCALE)  # image ground truth
				image_predicted = cv2.imread(path_predicted + '/' + frames_predicted_list[i], cv2.IMREAD_GRAYSCALE)

				# resize if needed:
				if image_gt.shape != image_predicted.shape:
					if image_gt.shape[0] * image_gt.shape[1] < image_predicted.shape[0] * image_predicted.shape[1]:
						new_shape = (image_gt.shape[1], image_gt.shape[0])
						image_predicted = cv2.resize(image_predicted, new_shape)
					else:
						new_shape = (image_predicted.shape[1], image_predicted.shape[0])
						image_gt = cv2.resize(image_gt, new_shape)

				# calculate metric
				corr_matrix = np.corrcoef(image_predicted.flatten(), image_gt.flatten())
				metric = corr_matrix[0, 1]
				#metric = CC(image_predicted.flatten(), image_gt.flatten())
				metric_list.append(metric)
				print('Correlation coefficient is: ', metric, 'of frame #: ', i + 1, ', video: ', dname)

			else:
				continue
				#print('There was not corresponding saliency and fixation map')

				# save information
				#with open(path_extra_info, 'a') as file:
					#print('There was not corresponding salience/fixation map for frame:', frames_predicted_list[i], 'in video:', dname, file=file)

				#continue

		# get statistics
		metric_series = pd.Series(metric_list)
		results = pd.DataFrame({dname:metric_series.describe()}) # dname is the name of the column
		results_list.append(results)

	# results_list is a list of dataframes that have the same index column(s)
	results_df = pd.concat(results_list, axis=1)

	# save the resulting dataframe as a CSV file
	results_df.to_csv(path_results + '/' + perf_metric + technique + '.csv', index=True)
	print('RESULTS SAVED!')




def CC_evelyn(image_gt, image_predicted):
	corr_matrix = np.corrcoef(image_gt.flatten(), image_predicted.flatten())
	corr_coefficient = corr_matrix[0,1]
	return  corr_coefficient



# Correlation coefficient from: https://github.com/imatge-upc/saliency-2019-SalBCE/blob/master/src/evaluation/metrics_functions.py (This way I get the same results as with my approach, but my approach is less coding)
def CC(saliency_map1, saliency_map2):
	'''
	Pearson's correlation coefficient between two different saliency maps
	(CC=0 for uncorrelated maps, CC=1 for perfect linear correlation).
	Parameters
	----------
	saliency_map1 : real-valued matrix
		If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
	saliency_map2 : real-valued matrix
	Returns
	-------
	CC : float, between [-1,1]
	'''
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have zero mean and unit std
	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')
	# Compute correlation coefficient
	return np.corrcoef(map1.ravel(), map2.ravel())[0,1]



def normalize(x, method='standard', axis=None):
	'''Normalizes the input with specified method.
	Parameters
	----------
	x : array-like
	method : string, optional
		Valid values for method are:
		- 'standard': mean=0, std=1
		- 'range': min=0, max=1
		- 'sum': sum=1
	axis : int, optional
		Axis perpendicular to which array is sliced and normalized.
		If None, array is flattened and normalized.
	Returns
	-------
	res : numpy.ndarray
		Normalized array.
	'''
	# TODO: Prevent divided by zero if the map is flat
	x = np.array(x, copy=False)
	if axis is not None:
		y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
		shape = np.ones(len(x.shape))
		shape[axis] = x.shape[axis]
		if method == 'standard':
			res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
		elif method == 'range':
			res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
		elif method == 'sum':
			res = x / np.float_(np.sum(y, axis=1).reshape(shape))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	else:
		if method == 'standard':
			res = (x - np.mean(x)) / np.std(x)
		elif method == 'range':
			res = (x - np.min(x)) / (np.max(x) - np.min(x))
		elif method == 'sum':
			res = x / float(np.sum(x))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	return res


if __name__ == '__main__':
    main()