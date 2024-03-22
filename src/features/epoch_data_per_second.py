import os
import pandas as pd
import json
import csv
import matplotlib.pyplot as plt

def epoch_data_into_seconds(config):
    for game in config['games']:
        for subject in config['subjects']:
            read_path = config['processed_data_path'] + game + '/combined/' 
            files = os.listdir(read_path)
            for file in files:
                sub = file[0:2]
                session = file[3:10]
                process_all_data_per_frame_path = config['processed_data_path'] + game + '/combined/' + sub + '_' + session + '_all_features_per_frame.csv'
                all_data = pd.read_csv(process_all_data_per_frame_path)

                time_stamps = all_data['time_stamps_game'].to_list()
                time_stamps_seconds = all_data['time_stamp_in_seconds'].to_list()
                action = all_data['action'].to_list()
                shift = all_data['shift'].to_list()
                time_stamps_eye = all_data['time_stamps_eye'].to_list()
                eye_x_pos = all_data['eye_x_pos'].to_list()
                eye_y_pos = all_data['eye_y_pos'].to_list()
                ProbDistraction = all_data['ProbDistraction'].to_list()
                ProbLowEng = all_data['ProbLowEng'].to_list()
                ProbHighEng = all_data['ProbHighEng'].to_list()
                ProbAveWorkload = all_data['ProbAveWorkload'].to_list()
                objects_x_start = all_data['objects_x_start'].to_list()
                objects_y_start = all_data['objects_y_start'].to_list()
                objects_width = all_data['objects_width'].to_list()
                objects_height = all_data['objects_height'].to_list()
                objects_centroid_x = all_data['objects_centroid_x'].to_list()
                objects_centroid_y = all_data['objects_centroid_y'].to_list()
                objects_area = all_data['objects_area'].to_list()

                last_second = 0
                current_second = config['epoch_length']
                eye_x_pos_s = []
                eye_y_pos_s = []
                time_stamps_eye_s = []
                time_stamps_s = []
                time_stamps_seconds_s = []
                action_s = []
                shift_s = []
                ProbDistraction_s = []
                ProbLowEng_s  = []
                ProbHighEng_s = []
                ProbAveWorkload_s  = []
                objects_x_start_s  = []
                objects_y_start_s = []
                objects_width_s  = []
                objects_height_s = []
                objects_centroid_x_s  = []
                objects_centroid_y_s  = []
                objects_area_s = []
                epoch_label = 0

                save_path = config['processed_data_path'] + game + '/epoched_data/' + sub + '_' +session + '_' + str(config['epoch_length']) +  'seconds_epoched_data.csv'
                with open(save_path, 'w') as file: 
                    dw = csv.DictWriter(file, delimiter=',',  
                                        fieldnames=config['header_list_all_data_per_second']) 
                    dw.writeheader() 


                for i in range(0,len(time_stamps_seconds)):
                    if time_stamps_seconds[i] > current_second:
                        last_second = current_second
                        current_second = current_second + config['epoch_length']

                        row = [epoch_label,time_stamps_s,time_stamps_seconds_s, action_s,shift_s,time_stamps_eye_s,eye_x_pos_s,eye_y_pos_s,ProbDistraction_s,ProbLowEng_s,ProbHighEng_s,ProbAveWorkload_s,objects_x_start_s,objects_y_start_s,objects_width_s,objects_height_s ,objects_centroid_x_s,objects_centroid_y_s ,objects_area_s]
                        with open(save_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)

                        epoch_label = epoch_label +1
                        eye_x_pos_s = []
                        eye_y_pos_s = []
                        time_stamps_eye_s = []
                        time_stamps_s = []
                        time_stamps_seconds_s = []
                        action_s = []
                        shift_s = []
                        ProbDistraction_s = []
                        ProbLowEng_s  = []
                        ProbHighEng_s = []
                        ProbAveWorkload_s  = []
                        objects_x_start_s  = []
                        objects_y_start_s = []
                        objects_width_s  = []
                        objects_height_s = []
                        objects_centroid_x_s  = []
                        objects_centroid_y_s  = []
                        objects_area_s = []

                    if time_stamps_seconds[i]  > last_second and time_stamps_seconds[i] < current_second:
                        eye_x_pos_s.extend(json.loads(eye_x_pos[i]))
                        eye_y_pos_s.extend(json.loads(eye_y_pos[i]))
                        time_stamps_eye_s.extend(json.loads(time_stamps_eye[i]))
                        action_s.append(action[i])
                        shift_s.append(shift[i])
                        time_stamps_s.append(time_stamps[i])
                        time_stamps_seconds_s.append(time_stamps_seconds[i])
                        ProbDistraction_s.append(ProbDistraction[i])
                        ProbLowEng_s.append(ProbLowEng[i])
                        ProbHighEng_s.append(ProbHighEng[i])
                        ProbAveWorkload_s.append(ProbAveWorkload[i])
                        objects_x_start_s.append(objects_x_start[i])
                        objects_y_start_s.append(objects_y_start[i])
                        objects_width_s.append(objects_width[i])
                        objects_height_s.append(objects_height[i])
                        objects_centroid_x_s.append(objects_centroid_x[i])
                        objects_centroid_y_s.append(objects_centroid_y[i])
                        objects_area_s.append(objects_area[i])

        return
        
    
