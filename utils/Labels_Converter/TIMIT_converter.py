# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np

#***************************************************************************
#
#***************************************************************************

def read_file(file_name):
    """
    Converts the text in a txt, txtgrid,... into a python list
    """
    f = open(file_name,'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n','')
    return lines

#-

def label_sig(phone_group,seq_length,time_table,step):    
    categories = list(phone_group)#Phonemes classes
    categories = categories[1:]#The first element is the column with the German phonemes
    labels = np.zeros([seq_length,len(categories)])
    for k in time_table:
        IPA = list(k.keys())[0]
        phone = IPA.split('_')[0]
        check_phone = phone_group[phone_group['Phone']==phone]#Look for the phoneme in the excel file
        if len(check_phone)>0:
            ti = int(k[IPA][0]/step)#Intial time inthe json file (Verbmobil)
            tf = int(k[IPA][1]/step)#Final time  inthe json file (Verbmobil)   
            labels[ti:tf,:] = check_phone[categories].values
    return labels

#***************************************************************************
#
#***************************************************************************

path_data = 'datasets/USC-TIMIT/rawdata_OneDrive_2_2025-4-15/MRI/Data'
list_subjects = os.listdir(path_data)
# list_subjects.remove('desktop.ini')

PH_Table = pd.read_excel('utils/TIMIT_MRI_Get_Phone_Alignment/Phonemic_Table.xlsx')
PH_Class = list(PH_Table)[1:]
fps = 15
step = float(1/fps) # Step time
#List of subject/folders
for sub in list_subjects:
    print()
    print(sub)
    path_trans = path_data+'/'+sub+'/trans'
    list_trans = os.listdir(path_trans)
    #Transcription files
    for tr in list_trans:
        file_name = tr.split('.')[0]
        # path_frames = './Data/'+sub+'/frames_15fps/'+folder_name
        # list_frames = os.listdir(path_frames)
        # list_frames.remove('desktop.ini')
        #-
        #Read time stamps from file
        #-
        path_file = path_trans+'/'+tr
        file = read_file(path_file)
        #Get number of frames
        Dur = float(file[-1:][0].split(',')[1])#duration of recording from last position of file
        num_frames = len(np.arange(0,Dur,step))
        #Label matrix
        labels = np.zeros([num_frames, len(PH_Class)], dtype=int)
        i = 0
        for frame in range(num_frames):
            frame_time = frame * step

            while i < len(file) and float(file[i].split(',')[1]) <= frame_time:
                i += 1

            if i >= len(file):
                break

            flist = file[i].split(',')
            ti = float(flist[0])  # Initial time of phoneme
            tf = float(flist[1])  # End time of phoneme
            phone = flist[2]      # Phoneme

            if not (ti <= frame_time < tf):
                continue

            check_phone = PH_Table[PH_Table['Phoneme'] == phone.upper()]
            if len(check_phone) > 0:
                labels[frame, :] = check_phone[PH_Class].values
            else:
                print()
                print('NOT FOUND', file_name + '_' + str(frame) + '.png', phone.upper())

        # for f in file:
        #     flist = f.split(',')
        #     ti = float(flist[0])#Initial time of phoneme
        #     tf = float(flist[1])#End time of phoneme
        #     phone = flist[2]#Phoneme
        #     #Look for the phoneme in the excel file
        #     check_phone = PH_Table[PH_Table['Phoneme']==phone.upper()]
        #     if len(check_phone)>0:
        #         ti_frame = int(ti/step)
        #         tf_frame = int(tf/step)
        #         labels[ti_frame:tf_frame,:] = check_phone[PH_Class].values
        #     else:
        #         print()
        #         print('NOT FOUND',file_name+'_'+str(i)+'.png',phone.upper())
        #     i+=1
        #Frame names
        Targets = pd.DataFrame()
        lnames = []
        for i in range(num_frames):
            lnames.append(file_name+'_'+str(i)+'.png')
        lnames = np.asarray(lnames).reshape(-1,1)
        df = np.hstack([lnames,labels.astype(int)])
        Targets = pd.DataFrame(df)
        cols = {0:'Filename'}
        inip = len(cols)
        for i in range(inip,len(PH_Class)+inip):
             cols[i] = PH_Class[i-inip]
        Targets = Targets.rename(columns=cols)
        path_save = 'datasets/labels_TIMIT/'+sub+'/frames_'+str(fps)+'fps'
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        Targets.to_csv(path_save+'/'+file_name+'.csv',index=False)
            
        
        
        