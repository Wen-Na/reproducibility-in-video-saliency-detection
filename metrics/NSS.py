import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re

def main():

    technique = '_tased' # DA or Base Model (HD)?
    perf_metric = 'NSS_results'
    separated_directories = True # there are cases where saliency directory contains two subdirectories: frames (sal. prediction) and images (sal. prediction overlayed on the original video frame)

    # set paths to saliency maps and fixation maps
    path_saliency_maps = '../predicted_saliency_maps_edu_dataset_jens_zhang_weights2nd'
    path_fixation_maps = '/nfs/home/navarretee/miniconda3/envs/vinet_env/vinet/eduVideos_dataset_jens_zhang'
    path_results = 'output_ev_metrics/DIEM17videos_dataset_weigths2nd' 	# set path where to store the output results
    if not os.path.isdir(path_results):
        os.makedirs(path_results)
    path_extra_info = path_results + '/extra_info_NSS_results_new.txt'

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
            path_predicted = os.path.join(path_saliency_maps, video_name, 'images')  # prediction/saliency maps
            #path_gt = os.path.join(path_fixation_maps, dname)  # ground truth/fixations maps
            path_gt = os.path.join(path_fixation_maps, dname, 'maps')  # ground truth/fixations maps
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
            if i + 1 > len(frames_predicted_list):
                break

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

                # calculate matric
                metric = NSS(image_predicted, image_gt)
                metric_list.append(metric)  # metric is a tensor, so to get the value inside we use item()
                print('NSS is: ', metric, 'of frame #: ', i + 1, ', experiment & video: ', dname)


        # get statistics
        metric_series = pd.Series(metric_list)
        results = pd.DataFrame({dname:metric_series.describe()}) # dname is the name of the column
        results_list.append(results)

    # results_list is a list of dataframes that have the same index column(s)
    results_df = pd.concat(results_list, axis=1)

    # save the resulting dataframe as a CSV file
    results_df.to_csv(path_results + '/' + perf_metric + technique + '_new.csv', index=True)
    print('RESULTS SAVED!')



def NSS(fixation_map, saliency_map):
    mean = saliency_map.mean()
    std = saliency_map.std()

    # Calculate the normalized saliency map using the formula (P - mean(P)) / std(P)
    normalized_saliency_map = (saliency_map - mean) / (std + 1e-5)  # Adding a small constant to avoid division by zero

    nss_value = (normalized_saliency_map * fixation_map).sum() / fixation_map.sum()

    return nss_value




if __name__ == '__main__':
    main()
