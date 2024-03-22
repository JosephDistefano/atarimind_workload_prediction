import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import numpy as np
import ast
import matplotlib

def visualize_epochs(config):
    for game in config['games']:
        game= 'breakout'
        for subject in config['subjects']:
            read_path = config['processed_data_path'] + game + '/game/' 
            files = os.listdir(read_path)
            for file in files:
                sub = file[0:2]
                session = file[3:10]
                all_data_path = config['processed_data_path'] + game + '/epoched_data/' + sub + '_' + session + '_1seconds_epoched_data.csv'
                frame_path = config['processed_data_path'] + game + '/frame/' + sub + '_' + session +'_'
                ccl_frame_path = config['processed_data_path'] + game + '/CCL_frames/' + sub + '_' + session +'_'
                eye_features_path = config['processed_features_path'] + game + '/gaze/' +  sub + '_' + session +'_' + 'gaze_features.csv'
                eye_feature_data = pd.read_csv(eye_features_path)
                fixations = eye_feature_data['fixations']
                fixation_stats = eye_feature_data['fixations_stats']
                saccades_stats = eye_feature_data['sacades_stats']
                saccades = eye_feature_data['sacades']
                all_data = pd.read_csv(all_data_path)
                epoch_label = all_data['epoch_label'].to_list()
                time_stamps = all_data['time_stamps_game'].to_list()
                action = all_data['action'].to_list()
                shift = all_data['shift'].to_list()
                x_pos = all_data['eye_x_pos'].to_list()
                y_pos = all_data['eye_y_pos'].to_list()
                ProbDistraction = all_data['ProbDistraction'].to_list()
                ProbLowEng = all_data['ProbLowEng'].to_list()
                ProbHighEng = all_data['ProbHighEng'].to_list()
                ProbAveWorkload = all_data['ProbAveWorkload'].to_list()
                for i in range(0,len(epoch_label)):
                    time_stamps_e = np.asarray([float(x) for x in time_stamps[i].strip('[]').split(', ')])
                    eye_gaze_x_e = np.asarray([float(x) for x in x_pos[i].strip('[]').split(', ')])
                    eye_gaze_y_e = np.asarray([float(x) for x in y_pos[i].strip('[]').split(', ')])
                    if fixations[i] != '[]':
                        data_list = ast.literal_eval(fixations[i])
                        fixations_e = [item for sublist in data_list for item in sublist]
                        data_list_2 = ast.literal_eval(fixation_stats[i])
                        fixations_stats_e = [[float(item) for item in sublist] for sublist in data_list_2]   
                        loc_fix_x = []      
                        loc_fix_y = []     
                        for value in fixations_stats_e:
                            loc_fix_x.append(value[-2])
                            loc_fix_y.append(value[-1])
                    else: 
                        fixations_e = []
                        loc_fix_x = []
                        loc_fix_y = []

                    if saccades[i] != '[]':
                        data_list = ast.literal_eval(saccades[i])
                        saccades_e = data_list
                        # saccades_e = [item for sublist in data_list for item in sublist]
                        data_list_2 = ast.literal_eval(saccades_stats[i])
                        sacade_stats_e = data_list_2
                        # sacade_stats_e = [[float(item) for item in sublist] for sublist in data_list_2]   
                        loc_sac_start_x = []      
                        loc_sac_start_y = []
                        loc_sac_end_x = []
                        loc_sac_end_y = []     
                        for value in sacade_stats_e:
                            loc_sac_start_x.append(value[-4])
                            loc_sac_start_y.append(value[-3])
                            loc_sac_end_x.append(value[-2])
                            loc_sac_end_y.append(value[-1])
                    else :
                        saccades_e = []
                        loc_sac_start_x = []      
                        loc_sac_start_y = []
                        loc_sac_end_x = []
                        loc_sac_end_y = []             

                    fig,axs = plt.subplots(2,len(time_stamps_e))
                    for i in range(0,len(time_stamps_e)):
                        frame_path_f = frame_path + str(time_stamps_e[i]) + '.png'
                        img = plt.imread(frame_path_f)
                        axs[0,i].imshow(img)

                    axs[1,0].scatter(eye_gaze_x_e,eye_gaze_y_e)
                    axs[1,0].set(ylim=(210,0))
                    axs[1,0].set(xlim=(0,160))
                    axs[1,0].set_xlabel('eye gaze')
                    axs[1,1].bar(0,len(fixations_e))
                    axs[1,1].set_xlabel('number of fixations')
                    axs[1,2].bar(0,len(saccades_e))
                    axs[1,2].set_xlabel('number of saccades')
                    axs[1,3].set(ylim=(210,0))
                    axs[1,3].set(xlim=(0,160))
                    axs[1,3].scatter(loc_fix_x,loc_fix_y)
                    axs[1,3].set_xlabel('fixation locations')
                    axs[1,4].set(ylim=(210,0))
                    axs[1,4].set(xlim=(0,160))
                    axs[1,4].set_xlabel('saccade locations')
                    print(loc_sac_start_x,loc_sac_start_y,len(saccades_e))
                    print(loc_sac_end_x,loc_sac_end_y)
                    axs[1,4].scatter(loc_sac_start_x,loc_sac_start_y, color='b')
                    axs[1,4].scatter(loc_sac_end_x,loc_sac_end_y,color ='k')


                
                    plt.waitforbuttonpress()    
                    plt.clf()
                    plt.close()
        #  
    #                 ccl_frame_path_f = ccl_frame_path + str(time_stamps[i]) + '.png'
    #                 x_pos_f = [float(x) for x in x_pos[i].strip('[]').split(', ')]
    #                 y_pos_f = [float(x) for x in y_pos[i].strip('[]').split(', ')]
    #                 ProbDistraction_f = ProbDistraction[i] 
    #                 ProbAveWorkload_f = ProbAveWorkload[i]
    #                 ProbHighEng_f = ProbHighEng[i]
    #                 img = plt.imread(frame_path_f)
    #                 cimg = plt.imread(ccl_frame_path_f)
    #                 axs[0].imshow(img)
    #                 axs[0].scatter(x_pos_f,y_pos_f,c='w', s=40)
    #                 act_shift = 'action: ' + str(action[i]) + '  shift: ' + str(shift[i])
    #                 dl = 'Distraction:  ' +str(ProbDistraction_f)
    #                 wl = 'Workload:  ' + str(ProbAveWorkload_f)
    #                 hel = 'High Engagament:  ' +str(ProbHighEng_f)
    #                 axs[0].set_xlabel(act_shift)
    #                 axs[1].bar(0,ProbDistraction_f)
    #                 axs[1].set_xlabel(dl)
    #                 axs[1].set(ylim=(0,1))
    #                 axs[2].bar(0,ProbAveWorkload_f)
    #                 axs[2].set_xlabel(wl)
    #                 axs[2].set(ylim=(0,1))
    #                 axs[3].bar(0,ProbHighEng_f)
    #                 axs[3].set_xlabel(hel)                    
    #                 axs[3].set(ylim=(0,1))
    #                 axs[4].imshow(cimg)
    #                 axs[4].set_xlabel('CCL')                    
    #                 plt.pause(.01)

    #                 axs[0].cla()
    #                 axs[1].cla()
    #                 axs[2].cla()
    #                 axs[3].cla()
    #                 axs[4].cla()
    # return 
    