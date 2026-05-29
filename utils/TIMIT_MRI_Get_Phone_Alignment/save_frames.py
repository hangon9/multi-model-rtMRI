import numpy as np
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torchaudio
# Ignore warnings
import warnings
import ffmpeg
import time
from sklearn.model_selection import train_test_split
from rich.progress import track
import subprocess, sys
from tqdm import tqdm
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from transformers import ViTImageProcessor
from scipy.io.wavfile import write
from moviepy.editor import *
# Define custom progress bar
progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

def create_folder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

warnings.filterwarnings("ignore")
#TODO: (1) Extract Video (2) Extract and Resample Audio (3) Non-Urgent --> Transcribe!
plt.ion()   # interactive mode
# def create_folder(new_folder):
#     if not os.path.exists(new_folder):
#         os.makedirs(new_folder)

def resample_videos(path_scr,path_out, fps =15):
    if not os.path.exists(path_out):
        command = "ffmpeg -i " + path_scr + " -r "+ str(fps) + " " + path_out
        os.system(command)

    #Other Possibilities
    #subprocess.run(command)#, shell=True, executable="/bin/bash")

    #Without Audio!
    #stream = ffmpeg.input(path_scr)
    #stream = stream.filter('fps', fps = fps, round = 'up')
    #stream = ffmpeg.output(stream, path_out)
    #ffmpeg.run(stream)

# def save_audios(path1, path2):
#     if not os.path.exists(path2[:-4]+'.wav'):
#         video_ = VideoFileClip(path1)  # 2.
#         audio_ = video_.audio  # 3.
#         #audio_.write_audiofile(path2[:-4]+'wav')  # 4.#
#         audio_.write_audiofile(path2[:-4]+'.wav', codec='pcm_s16le')
import ffmpeg
def save_audios(path1, path2):


    # Load the video file
    input_file = ffmpeg.input(path1)

    # Extract the audio and save it as an MP3 file
    input_file.output(path2[:-4]+'.wav', acodec='pcm_s16le').run()
import cv2
import os
def save_images(path1, path2, file):


    # Read the video from specified path
    cam = cv2.VideoCapture(path1)
    if not os.path.exists(path2[:-4] ):
        try:

            # creating a folder named data
            if not os.path.exists(path2[:-4]):
                os.makedirs(path2[:-4])

                # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')

            # frame
        currentframe = 0

        while (True):

            # reading from frame
            ret, frame = cam.read()

            if ret:
                # if video is still left continue creating images
                name = path2[:-4] + '/'+ file[:-4]+ '_'+ str(
                    currentframe) + '.png'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()


#Saving Images and Audios!



gender = ['F', 'M']

for g in gender:
        for i in range(5):
            
            path_scr=r"C:\Users\tomas\Documents\Experiments\TSD_2025_Daiqi\USC-TIMIT\MRI\Data/"+g+str(i+1)+"/avi/"
            path_out = r"C:\Users\tomas\Documents\Experiments\TSD_2025_Daiqi\USC-TIMIT\MRI\Data/"+g+str(i+1)+"/avi_15fps/"
            
            files = sorted(os.listdir(path_scr))
            create_folder(path_out)


            for file in track(files, description="Processing..."):


                #Resample
                resample_videos(os.path.join(path_scr, file), os.path.join(path_out, file))
                #save_audios(os.path.join(path_scr, file, 'videos', vid), os.path.join(path_out, file, 'audios', vid))
                # if not os.path.exists(os.path.join(path_out, file)):
                #     save_images(os.path.join(path_scr, file), os.path.join(path_out, file), file)
                #load_video(os.path.join(path_out, file, 'videos', vid), os.path.join(path_scr, file), vid)
                #resample_videos(os.path.join(path_scr, file,tmp,vid), os.path.join(path_out, file, 'videos', vid))



gender = ['F', 'M']

for g in gender:
        for i in range(5):
            
            path_scr=r"C:\Users\tomas\Documents\Experiments\TSD_2025_Daiqi\USC-TIMIT\MRI\Data/"+g+str(i+1)+"/avi_15fps/"
            path_out = r"C:\Users\tomas\Documents\Experiments\TSD_2025_Daiqi\USC-TIMIT\MRI\Data/"+g+str(i+1)+"/frames_15fps/"
            
            files = sorted(os.listdir(path_scr))
            create_folder(path_out)


            for file in track(files, description="Processing..."):


                #Resample
                #resample_videos(os.path.join(path_scr, file), os.path.join(path_out, file))
                #save_audios(os.path.join(path_scr, file, 'videos', vid), os.path.join(path_out, file, 'audios', vid))
                if not os.path.exists(os.path.join(path_out, file)):
                    save_images(os.path.join(path_scr, file), os.path.join(path_out, file), file)
                #load_video(os.path.join(path_out, file, 'videos', vid), os.path.join(path_scr, file), vid)
                #resample_videos(os.path.join(path_scr, file,tmp,vid), os.path.join(path_out, file, 'videos', vid))