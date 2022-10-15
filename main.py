import pickle
from util import *
import cv2 
import mediapipe as mp
import numpy as np
import os
import csv
from tqdm import tqdm
import pandas as pd
from sklearn.svm import SVC
import warnings
import tkinter as tk
import customtkinter as ck
from PIL import Image, ImageTk
warnings.filterwarnings('ignore')



window = tk.Tk()
window.geometry("480x640")
window.title('AIYOGA')
ck.set_appearance_mode("Dark")
classBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0') 

frame = tk.Frame(height=1080, width=1920)
frame.place(x=10, y=90) 
lmain = tk.Label(frame) 
lmain.place(x=0, y=0) 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True, model_complexity = 2, min_detection_confidence = 0.5
) 

with open('modelRF.sav', 'rb') as f:
        loaded_model = pickle.load(f)

cap = cv2.VideoCapture('testvid.mp4')
# IMAGE_FILES = ['48.jpg']
classdec = ''
def detect():

    ret,frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    mp_drawing.draw_landmarks(image,
                                results.pose_landmarks, 
                                mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), #joint
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) #bone
    image.flags.writeable = True
    image_height, image_width, _ = image.shape
    try :
        landmarks = results.pose_landmarks.landmark
        keypoints = getLandmarks(landmarks)
        angles = getAngles(keypoints)
        anglesL = list(angles.values())
        prob = loaded_model.predict_proba([anglesL])
        classyoga = loaded_model.predict([anglesL])
        
        # print(prob, classyoga )
        
    
    except Exception as e:
        print(e)
    

    
    img = image[:,:460,:]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)
    classBox.configure(text=classyoga[0])



detect()
window.mainloop()


