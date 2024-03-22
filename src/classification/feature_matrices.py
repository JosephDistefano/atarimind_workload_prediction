import os
import pandas as pd
import numpy as np
import ast
import csv

def create_feature_matrix_eye_ccl(config):
    for game in config['games']:
        game_feature_matrices_path = config['processed_features_path'] + game + '/feature_matrices/' + 'eye_ccl_feature_matrix.csv'
        with open(game_feature_matrices_path, 'w') as file: 
            dw = csv.DictWriter(file, delimiter=',',fieldnames=config['header_eye_ccl_feature_matrix']) 
            dw.writeheader() 
        for subject in config['subjects']:
            read_path = config['processed_data_path'] + game + '/game/' 
            files = os.listdir(read_path)
            for file in files:
                sub = file[0:2]
                session = file[3:10]
                
                all_data_path = config['processed_data_path'] + game + '/epoched_data/' + sub + '_' + session + '_1seconds_epoched_data.csv'
                all_data = pd.read_csv(all_data_path)
                epoch_label = all_data['epoch_label'].to_list()
                time_stamps = all_data['time_stamps_game'].to_list()
                ProbDistraction = all_data['ProbDistraction'].to_list()
                ProbLowEng = all_data['ProbLowEng'].to_list()
                ProbHighEng = all_data['ProbHighEng'].to_list()
                ProbAveWorkload = all_data['ProbAveWorkload'].to_list()
                eeg_epoch_label = all_data['epoch_label']

                eye_features_path = config['processed_features_path'] + game + '/gaze/' +  sub + '_' + session +'_' + 'gaze_features.csv'
                eye_feature_data = pd.read_csv(eye_features_path)
                fixations = eye_feature_data['fixations']
                fixation_stats = eye_feature_data['fixations_stats']
                saccades_stats = eye_feature_data['sacades_stats']
                saccades = eye_feature_data['sacades']
                eye_epoch_label = eye_feature_data['epoch']
                
                ccl_features_path = config['processed_features_path'] + game + '/ccl/' +  sub + '_' + session +'_' + 'ccl_features.csv'
                ccl_features_data = pd.read_csv(ccl_features_path)
                mean_num_groups_s = ccl_features_data["mean_num_groups_s"]
                max_num_groups_s = ccl_features_data["max_num_groups_s"]
                min_num_groups_s = ccl_features_data["min_num_groups_s"]
                mean_size_groups_s = ccl_features_data["mean_size_groups_s"]
                max_size_groups_s = ccl_features_data["max_size_groups_s"]
                min_size_groups_s = ccl_features_data["min_size_groups_s"]
                mean_num_groups_d = ccl_features_data["mean_num_groups_d"]
                max_num_groups_d = ccl_features_data["max_num_groups_d"]
                min_num_groups_d = ccl_features_data["min_num_groups_d"]
                mean_size_groups_d = ccl_features_data["mean_size_groups_d"]
                max_size_groups_d = ccl_features_data["max_size_groups_d"]
                min_size_groups_d = ccl_features_data["min_size_groups_d"]
                ccl_epoch_label = ccl_features_data['epoch']

                for i in range(0,len(epoch_label)-1):
                    print(eye_epoch_label[i],ccl_epoch_label[i],eeg_epoch_label[i])
                    if epoch_label[i] not in [81,82]:
                        if fixations[i] != '[]':
                            data_list = ast.literal_eval(fixations[i])
                            fixations_e = len([item for sublist in data_list for item in sublist])
                        else: 
                            fixations_e = 0
                        if saccades[i] != '[]':
                            saccades_e = len(ast.literal_eval(saccades[i]))        
                        else :
                            saccades_e = 0
                        if ProbDistraction[i] != '[]':
                            ProbDistraction_e = [float(x) for x in ProbDistraction[i].strip('[]').split(',')][0]
                        else:
                            ProbDistraction_e = 0
                        if ProbLowEng[i] != '[]':
                            ProbLowEng_e =  [float(x) for x in ProbLowEng[i].strip('[]').split(',')][0]
                        else:
                            ProbLowEng_e = 0
                        if ProbHighEng[i] != '[]':
                            ProbHighEng_e =  [float(x) for x in ProbHighEng[i].strip('[]').split(',')][0]
                        else:
                            ProbHighEng_e = 0
                        if ProbAveWorkload[i] != '[]':
                            ProbAveWorkload_e =  [float(x) for x in ProbAveWorkload[i].strip('[]').split(',')][0]                    
                        else:
                            ProbAveWorkload_e  = 0
                        row = [fixations_e,saccades_e,mean_num_groups_s[i],max_num_groups_s[i],mean_num_groups_s[i],max_num_groups_s[i],min_num_groups_s[i],mean_size_groups_s[i],max_size_groups_s[i],min_size_groups_s[i],mean_num_groups_d[i],max_num_groups_d[i],min_num_groups_d[i],mean_size_groups_d[i],max_size_groups_d[i],min_size_groups_d[i],ProbDistraction_e,ProbLowEng_e,ProbHighEng_e,ProbAveWorkload_e]
                        print(row)
                        with open(game_feature_matrices_path, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)

    return