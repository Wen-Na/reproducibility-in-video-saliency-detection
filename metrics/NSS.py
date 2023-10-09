import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re


def main():

    technique = '_ViNet' # approach
    perf_metric = 'NSS'
    separated_directories = True  # when maps are in separated directories

    # set paths to saliency maps and fixation maps. Also path to store the output results
    path_saliency_maps = 'path/to/saliency/maps'
    path_fixation_maps = 'path/to/fixation/maps'
    path_results = 'path/to/save/results'
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

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
            path_gt = os.path.join(path_fixation_maps, dname, 'maps')  # ground truth/fixations maps
        else:
            # path to video frames
            path_predicted = os.path.join(path_saliency_maps, video_name)  # prediction/saliency maps
            path_gt = os.path.join(path_fixation_maps, dname)  # ground truth/fixations maps

        # list of frames ground truth
        frames_gt_list = [f for f in listdir(path_gt) if isfile(join(path_gt, f))]
        frames_gt_list = [val for val in frames_gt_list if (val.endswith(".png") or val.endswith(".jpg"))]
        frames_gt_list.sort()

        # list of frames predicted
        frames_predicted_list = [f for f in listdir(path_predicted) if isfile(join(path_predicted, f))]
        frames_predicted_list = [val for val in frames_predicted_list if (val.endswith(".png") or val.endswith(".jpg"))]
        frames_predicted_list.sort()


        # compute metric
        metric_list = []
        for i in range(len(frames_gt_list)):

            # make sure that that background frame has corresponding salience/fixation map
            frame_number_reg_exp_gt = re.match(r'.*?(\d+).*', frames_gt_list[i])  # regular expression
            frame_number_gt = str(int(frame_number_reg_exp_gt.group(1)))  # extract reg exp and make it integer

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
                metric_list.append(metric)
                print('NSS is: ', metric, 'of frame #: ', i + 1, ', experiment & video: ', dname)


        # get statistics
        metric_series = pd.Series(metric_list)
        results = pd.DataFrame({dname:metric_series.describe()})
        results_list.append(results)

    # results as dataframe
    results_df = pd.concat(results_list, axis=1)

    # save the resulting dataframe as a CSV file
    results_df.to_csv(path_results + '/' + perf_metric + technique + '_new.csv', index=True)
    print('RESULTS SAVED!')



def NSS(fixation_map, saliency_map):

    # compute mean ans std
    mean = saliency_map.mean()
    std = saliency_map.std()

    # calculate the normalized saliency map
    normalized_saliency_map = (saliency_map - mean) / (std + 1e-5)  # Adding a small constant to avoid division by zero
    nss_value = (normalized_saliency_map * fixation_map).sum() / fixation_map.sum()

    return nss_value



if __name__ == '__main__':
    main()
