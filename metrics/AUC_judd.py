import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import cv2
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re


def main():

    technique = '_dsl' # DA or Base Model (HD)?
    perf_metric = 'AUCjudd_results'
    separated_directories = True # there are cases where saliency directory contains two subdirectories: frames (sal. prediction) and images (sal. prediction overlayed on the original video frame)

    # set paths to saliency maps and fixation maps
    path_saliency_maps = '../predicted_sal_maps_DIEM17videos_dataset_grey_scale'
    path_fixation_maps = '/nfs/home/navarretee/miniconda3/envs/vinet_env/vinet/DIEM_testset_17videos'
    path_results = 'output_ev_metrics/DIEM17videos_dataset_weigths3rd' # set path where to store the output results
    if not os.path.isdir(path_results):
        os.makedirs(path_results)
    path_extra_info = path_results + '/extra_info_AUCjudd_results.txt'

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
            frame_number_reg_exp_gt = re.match(r'.*?(\d+).*', frames_gt_list[i]) # regular expression
            frame_number_gt = str(int(frame_number_reg_exp_gt.group(1))) # extract reg exp and make it integer

            #if any(frame_number_gt == str(int(file.split('.')[0])) for file in frames_predicted_list):
            if any(frame_number_gt == str(int(re.match(r'.*?(\d+).*', file).group(1))) for file in frames_predicted_list):
                # upload images
                image_gt = cv2.imread(path_gt + '/' + frames_gt_list[i], cv2.IMREAD_GRAYSCALE)  # image ground truth
                image_predicted = cv2.imread(path_predicted + '/' + frames_predicted_list[i], cv2.IMREAD_GRAYSCALE)

                # resize image if needed:
                if image_gt.shape != image_predicted.shape:
                    if image_gt.shape[0] * image_gt.shape[1] < image_predicted.shape[0] * image_predicted.shape[1]:
                        new_shape = (image_gt.shape[1], image_gt.shape[0])
                        image_predicted = cv2.resize(image_predicted, new_shape)
                    else:
                        new_shape = (image_predicted.shape[1], image_predicted.shape[0])
                        image_gt = cv2.resize(image_gt, new_shape)

                # calculate matric
                metric = AUC_Judd_gpt(image_predicted, image_gt)
                metric_list.append(metric)  # metric is a tensor, so to get the value inside we use item()
                print('AUC is: ', metric, 'of frame #: ', i + 1, ', experiment & video: ', dname)


        # get statistics
        metric_series = pd.Series(metric_list)
        results = pd.DataFrame({dname:metric_series.describe()}) # dname is the name of the column
        results_list.append(results)

    # results_list is a list of dataframes that have the same index column(s)
    results_df = pd.concat(results_list, axis=1)

    # save the resulting dataframe as a CSV file
    results_df.to_csv(path_results + '/' + perf_metric + technique + '.csv', index=True)
    print('RESULTS SAVED!')



def AUC_Judd(saliencyMap, fixationMap, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap
    new_size = np.shape(fixationMap)
    if not np.shape(saliencyMap) == np.shape(fixationMap):
        #from scipy.misc import imresize
        new_size = np.shape(fixationMap)
        np.array(Image.fromarray(saliencyMap).resize((new_size[1], new_size[0])))

        #saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score



# A different code but should work (originally in matlab and changed with gpt to python
# From: https://github.com/cvzoya/saliency/tree/master/code_forMetrics
def AUC_Judd_gpt(saliencyMap, fixationMap, jitter=1, toPlot=0):
    """
    saliencyMap is the saliency map
    fixationMap is the human fixation map (binary matrix)
    jitter = 1 will add tiny non-zero random constant to all map locations
             to ensure ROC can be calculated robustly (to avoid uniform region)
    if toPlot=1, displays ROC curve
    """
    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('no fixationMap')
        return np.nan, None, None, None

    # make the saliencyMap the size of the image of fixationMap
    if saliencyMap.shape[0] != fixationMap.shape[0] or saliencyMap.shape[1] != fixationMap.shape[1]:
        saliencyMap = cv2.resize(saliencyMap, (fixationMap.shape[1], fixationMap.shape[0]))

    # jitter saliency maps that come from saliency models that have a lot of zero values
    if jitter:
        # jitter the saliency map slightly to disturb ties of the same numbers
        #saliencyMap += np.random.rand(*saliencyMap.shape) / 10000000
        saliencyMap = saliencyMap + np.random.rand(*saliencyMap.shape) / 10000000

    # normalize saliency map
    saliencyMap = (saliencyMap - np.min(saliencyMap)) / (np.max(saliencyMap) - np.min(saliencyMap))

    if np.isnan(np.sum(saliencyMap)):
        print('NaN saliencyMap')
        return np.nan, None, None, None

    S = saliencyMap.ravel()
    F = fixationMap.ravel()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = np.sort(Sth)[::-1]  # sort sal map values, to sweep through values
    tp = np.zeros(Nfixations + 2)
    fp = np.zeros(Nfixations + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = np.sum(S >= thresh)  # total number of sal map values above threshold
        tp[i + 1] = (i + 1) / Nfixations  # ratio sal map values at fixation locations above threshold
        fp[i + 1] = (aboveth - (i + 1)) / (Npixels - Nfixations)  # ratio other sal map values above threshold

    score = np.trapz(tp, fp)
    allthreshes = np.concatenate(([1], allthreshes, [0]))

    if toPlot:
        plt.subplot(121)
        plt.imshow(saliencyMap, cmap='gray')
        plt.title('SaliencyMap with fixations to be predicted')
        y, x = np.where(fixationMap)
        plt.plot(x, y, '.r')
        plt.subplot(122)
        plt.plot(fp, tp, '.b-')
        plt.title('Area under ROC curve: {:.4f}'.format(score))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()

    return score


if __name__ == '__main__':
    main()
