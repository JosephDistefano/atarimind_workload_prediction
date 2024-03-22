import os
import pandas as pd
import csv
from pygazeanalyser.detectors import fixation_detection,saccade_detection,calculate_saccades
import numpy as np
import matplotlib.pyplot as plt

def calculate_eye_features(config):
   for game in config['games']:
        for subject in config['subjects']:
            read_path = config['processed_data_path'] + game + '/epoched_data/' 
            files = os.listdir(read_path)
            for file in files:
                sub = file[0:2]
                session = file[3:10]
                all_data = pd.read_csv(read_path+file)
                save_gaze_features_path = config['processed_features_path'] + game +'/gaze/' + sub + '_' + session + '_' + 'gaze_features.csv'
                with open(save_gaze_features_path, 'w') as file: 
                    dw = csv.DictWriter(file, delimiter=',',fieldnames=config['header_gaze_features']) 
                    dw.writeheader() 

                epoch_label = all_data['epoch_label'].to_list()
                
                time_stamps = all_data['time_stamps_game'].to_list()
                
                eye_gaze_x = all_data['eye_x_pos'].to_list()
                eye_gaze_y = all_data['eye_y_pos'].to_list()
                eye_gaze_time_stamps = all_data['time_stamps_eye'].to_list()
                for i in range(0,len(epoch_label)):
                    if time_stamps[i] != '[]':
                        time_stamps_e = np.asarray([float(x) for x in time_stamps[i].strip('[]').split(', ')])
                        eye_gaze_time_stamps_e = np.asarray([float(x) for x in eye_gaze_time_stamps[i].strip('[]').split(', ')])
                        eye_gaze_time_stamps_e = (eye_gaze_time_stamps_e - eye_gaze_time_stamps_e[0])*1000
                        eye_gaze_x_e = np.asarray([float(x) for x in eye_gaze_x[i].strip('[]').split(', ')])
                        eye_gaze_y_e = np.asarray([float(x) for x in eye_gaze_y[i].strip('[]').split(', ')])
                        Sfix, Efix = fixation_detection(eye_gaze_x_e,eye_gaze_y_e,eye_gaze_time_stamps_e,missing=0.)
                        Ssac,Esac = calculate_saccades(eye_gaze_x_e,eye_gaze_y_e,eye_gaze_time_stamps_e,missing=0.)
                        row = [epoch_label[i],Sfix,Efix,Ssac,Esac]
                        with open(save_gaze_features_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)
                    if time_stamps[i] == '[]':    
                        row = [epoch_label[i],0,0,0,0]
                        with open(save_gaze_features_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)


                return

