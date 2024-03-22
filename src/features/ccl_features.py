import os
import pandas as pd
import numpy as np
import ast
import cv2
import csv
import matplotlib.pyplot as plt

def calculate_CCL_features(config):
   for game in config['games']:
        for subject in config['subjects']:
            read_path = config['processed_data_path'] + game + '/epoched_data/' 
            files = os.listdir(read_path)
            for file in files:
                sub = file[0:2]
                session = file[3:10]
                all_data = pd.read_csv(read_path+file)
                save_ccl_features_path = config['processed_features_path'] + game +'/ccl/' + sub + '_' + session + '_' + 'ccl_features.csv'
                with open(save_ccl_features_path, 'w') as file: 
                    dw = csv.DictWriter(file, delimiter=',',fieldnames=config['header_ccl_features']) 
                    dw.writeheader() 

                epoch_label = all_data['epoch_label'].to_list()
                
                time_stamps = all_data['time_stamps_game'].to_list()

                # frame_path = config['processed_data_path'] + game + '/frame/' + sub + '_' + session +'_'

                # for i in range(0,len(time_stamps)):
                #     if time_stamps[i] != '[]':
                #         temp = [float(x) for x in time_stamps[i].strip('[]').split(', ')]          
                #         for j in range(0,len(temp)):
                #             plt.imread(frame_path+str(temp[j])+ '.png')

                time_stamps_seconds = all_data['time_stamp_in_seconds'].to_list()
                
                objects_x_start = all_data['objects_x_start'].to_list()
                objects_y_start = all_data['objects_y_start'].to_list()
                objects_width = all_data['objects_width'].to_list()
                objects_height = all_data['objects_height'].to_list()
                objects_centroid_x = all_data['objects_centroid_x'].to_list()
                objects_centroid_y = all_data['objects_centroid_y'].to_list()
                objects_area = all_data['objects_area'].to_list()

                for i in range(0,len(epoch_label)):
                    if time_stamps[i] != '[]':
                        time_stamps_e = [float(x) for x in time_stamps[i].strip('[]').split(', ')]
                        objects_x_start_e =  [list(map(float, s.strip('[]').split(', '))) for s in ast.literal_eval(objects_x_start[i])]
                        objects_y_start_e =  [list(map(float, s.strip('[]').split(', '))) for s in ast.literal_eval(objects_y_start[i])]
                        objects_width_e =  [list(map(float, s.strip('[]').split(', '))) for s in ast.literal_eval(objects_width[i])]
                        objects_height_e =  [list(map(float, s.strip('[]').split(', '))) for s in ast.literal_eval(objects_height[i])]
                        objects_centroid_x_e =  [list(map(float, s.strip('[]').split(', '))) for s in ast.literal_eval(objects_centroid_x[i])]
                        objects_centroid_y_e =  [list(map(float, s.strip('[]').split(', '))) for s in ast.literal_eval(objects_centroid_y[i])]
                        objects_area_e =  [list(map(float, s.strip('[]').split(', '))) for s in ast.literal_eval(objects_area[i])]


                        mean_num_groups_s,max_num_groups_s,min_num_groups_s,mean_size_groups_s,max_size_groups_s,min_size_groups_s = calculate_static_features(objects_area_e)
                        mean_num_groups_d,max_num_groups_d,min_num_groups_d,mean_size_groups_d,max_size_groups_d,min_size_groups_d = calculate_dynamic_features(config,time_stamps_e,game,sub,session)
                        row = [epoch_label[i],mean_num_groups_s,max_num_groups_s,min_num_groups_s,mean_size_groups_s,max_size_groups_s,min_size_groups_s,mean_num_groups_d,max_num_groups_d,min_num_groups_d,mean_size_groups_d,max_size_groups_d,min_size_groups_d]
                        with open(save_ccl_features_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)
                    else:
                        row = [epoch_label[i],0,0,0,0,0,0,0,0,0,0,0]
                        with open(save_ccl_features_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)

                return


def calculate_static_features(objects_area):

    epoch_objects = []
    epoch_areas = []

    for i in objects_area:
        epoch_objects.append(len(i))
        for j in i:
            epoch_areas.append(j)
    
    if len(epoch_objects)>1:
        mean_num_groups_s = np.mean(epoch_objects)
        max_num_groups_s = np.max(epoch_objects)
        min_num_groups_s = np.min(epoch_objects)
        # std_num_groups = np.std(epoch_objects)

        mean_size_groups_s = np.mean(epoch_areas)
        max_size_groups_s = np.max(epoch_areas)
        min_size_groups_s = np.min(epoch_areas)
        # std_size_groups = np.std(epoch_areas)

        # std_size_groups = 0
    elif len(epoch_objects)==1:
        mean_num_groups_s = epoch_objects[0]
        max_num_groups_s = epoch_objects[0]
        min_num_groups_s = epoch_objects[0]
        # std_num_groups = 0
        mean_size_groups_s = epoch_areas[0]
        max_size_groups_s = epoch_areas[0]
        min_size_groups_s = epoch_areas[0]
        # std_size_groups = 0
    else:
        mean_num_groups_s = 0
        max_num_groups_s = 0
        min_num_groups_s = 0
        # std_num_groups = 0
        mean_size_groups_s = 0
        max_size_groups_s = 0
        min_size_groups_s = 0

    

    return mean_num_groups_s,max_num_groups_s,min_num_groups_s,mean_size_groups_s,max_size_groups_s,min_size_groups_s

def calculate_dynamic_features(config,time_stamps_e,game,sub,session):
    processed_data_read_path = config['processed_data_path']
    frame_list=[]
    for time in time_stamps_e:
        frame_path = processed_data_read_path + game + '/frame/' + sub + '_' + session + '_' + str(time) + '.png'
        frame=cv2.imread(frame_path)
        if len((np.shape(cv2.imread(frame_path))))<3:
            continue
        img=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img , 8 , cv2.CV_32S)
        img = np.array(labels*255,dtype='uint8')
        frame_list.append(img)

        

    FrameDiffSet = []    
    frame_list = np.array(frame_list, dtype=np.float64)
    if len(frame_list) == 2:
        CurrentFrame = frame_list[1]
        LastFrame =frame_list[0]
        diffy = np.where(CurrentFrame-LastFrame>0,255,0)
        FrameDiffSet.append(np.where(diffy>0,255,0))
    else:
        for ind in range(1,len(frame_list)):
            CurrentFrame = frame_list[ind]
            LastFrame =frame_list[ind-1]
            diffy = CurrentFrame-LastFrame
            FrameDiffSet.append(np.where(diffy>0,255,0))

    dyn_areas = []
    kernel=np.ones((5,5),np.float64)/25
    for frame in FrameDiffSet:
        DifferenceFrame = cv2.filter2D(np.array(frame,dtype='uint8'),-1,kernel)
        _, DifferenceFrame = cv2.threshold(DifferenceFrame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(DifferenceFrame , 8 , cv2.CV_32S)
        RowToDelete = (np.where(stats[:,4]==np.max(stats[:,4]))[0]) # deleting background group
        stats = np.delete(stats,RowToDelete,0)
        num_labels-=1
        try:
            for area in stats[:,-1]:
                dyn_areas.append(area)
        except:
            dyn_areas.append(stats[-1])

    if len(dyn_areas)>1:
        mean_num_groups_d = np.mean(len(dyn_areas))
        max_num_groups_d = np.max(len(dyn_areas))
        min_num_groups_d = np.min(len(dyn_areas))
        # std_num_groups = np.std(dyn_areas)

        mean_size_groups_d = np.mean(dyn_areas)
        max_size_groups_d = np.max(dyn_areas)
        min_size_groups_d = np.min(dyn_areas)
        # std_size_groups = np.std(epoch_areas)
    elif len(dyn_areas)==1:
        mean_num_groups_d = len(dyn_areas)
        max_num_groups_d = len(dyn_areas)
        min_num_groups_d = len(dyn_areas)
        # std_num_groups = 0
        mean_size_groups_d = dyn_areas[0]
        max_size_groups_d = dyn_areas[0]
        min_size_groups_d = dyn_areas[0]
        # std_size_groups = 0
    else:
        mean_num_groups_d = 0
        max_num_groups_d = 0
        min_num_groups_d = 0
        # std_num_groups = 0
        mean_size_groups_d = 0
        max_size_groups_d = 0
        min_size_groups_d = 0



    return mean_num_groups_d,max_num_groups_d,min_num_groups_d,mean_size_groups_d,max_size_groups_d,min_size_groups_d