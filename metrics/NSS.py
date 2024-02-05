import cv2
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re
import torch
from PIL import Image
import torchvision.transforms as transforms


def main():

    technique = '_tmfi' # DA or Base Model (HD)?
    perf_metric = 'NSS'
    separated_directories = True # when maps are in separated directories
 
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

    # to upload images with library PIL
    transform = transforms.Compose([
    transforms.Grayscale(),  
    transforms.ToTensor(),  
    ])


    # loop through each folder and save results
    results_list = []
    for dname in list_fixation_maps:

        video_name = dname

        if separated_directories:
            # path to video frames
            path_predicted = os.path.join(path_saliency_maps, video_name, 'frames')  # prediction/saliency maps
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


        # compute metric
        metric_list = []
        for i in range(len(frames_gt_list)):

            # sometimes # of ground truth (gt/fixation maps) are more than saliency/fixation maps (because participants were longer in the computer), so, detect this and finish the loop
            if i + 1 > len(frames_predicted_list):
                break

            # make sure that that background frame has corresponding salience/fixation map
            frame_number_reg_exp_gt = re.match(r'.*?(\d+).*', frames_gt_list[i])  # regular expression
            frame_number_gt = str(int(frame_number_reg_exp_gt.group(1)))  # extract reg exp and make it integer
            
            if any(frame_number_gt == str(int(re.match(r'.*?(\d+).*', file).group(1))) for file in frames_predicted_list):
                # upload images
                path_gt_image = os.path.join(path_gt, frames_gt_list[i]) 
                image_gt = Image.open(path_gt_image).convert('L') # load with PIL
                image_gt = transform(image_gt)
  
                path_predicted_image = os.path.join(path_predicted, frames_predicted_list[i])
                image_predicted = Image.open(path_predicted_image).convert('L') # load with PIL
                image_predicted = transform(image_predicted)

                # resize if needed
                if image_gt.size() != image_predicted.size():
                    image_predicted = transforms.functional.resize(image_predicted, image_gt.shape[1:])

                # calculate matric
                metric = NSS(image_gt, image_predicted)
                metric = metric.item() 
                metric_list.append(metric)  
                print('NSS is: ', metric, 'of frame #: ', i + 1, ', experiment & video: ', dname)

        # get statistics
        metric_series = pd.Series(metric_list)
        results = pd.DataFrame({dname:metric_series.describe()}) # dname is the name of the column
        results_list.append(results)

    # results as dataframe
    results_df = pd.concat(results_list, axis=1)

    # save the resulting dataframe as a CSV file
    results_df.to_csv(path_results + '/' + perf_metric + technique + '.csv', index=True)

    print('RESULTS SAVED!')



def NSS(gt, s_map): # gt is the binary fixation map
    
    if s_map.size() != gt.size():
        s_map = s_map.cpu().squeeze(0).numpy()
        s_map = torch.FloatTensor(cv2.resize(s_map, (gt.size(2), gt.size(1)))).unsqueeze(0)
        s_map = s_map.cuda()
        gt = gt.cuda()
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)
    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    eps = 2.2204e-16
    s_map = (s_map - mean_s_map) / (std_s_map + eps)

    s_map = torch.sum((s_map * gt).view(batch_size, -1), 1)
    count = torch.sum(gt.view(batch_size, -1), 1)
    return torch.mean(s_map / count)



if __name__ == '__main__':
    main()
