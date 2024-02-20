# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:15:10 2020

@author: Samia
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:37:02 2019
@author: Samia
"""
import cv2
import numpy as np
import math
#import matplotlib.pyplot as plt
import time

def make_coordinate(image, line_parameters):   
    for line in line_parameters:
        slope, intercept = line
        y1= image.shape[0]
        #x= image.shape[1]
        y2 = int(y1*(.6))
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        """
    #print(abs(x1-x2))
    if slope<0:
        lx1 = x1
    if slope>0:
        rx1= x1
    #if x1>x/2:
        #q= x1
    d = abs(lx1-rx1)
    print(d)
    """
    return np.array([x1 ,y1, x2, y2])

def filtering(image, lines):
    l_candidate_array = []
    r_candidate_array = []
    left_lane = []
    right_lane = []
    left_lane_points = []
    right_lane_points =[]
    l_len = []
    r_len = []
    w = image.shape[1]
    z = 5.7
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        #print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        angle= int(math.atan(slope)*57.2958)
        l = math.sqrt((x1-x2)**2+(y1-y2)**2)
        #print(angle)
        if angle >0:
            angle = 180-angle
        else:
            angle = abs(angle)
        #print(angle)
        if angle in range(25,65):
            l_candidate_array.append((slope,intercept,l))
            l_len.append(l)
        if angle in range(110,155):  
            r_candidate_array.append((slope,intercept,l))
            r_len.append(l)
            
    #print('left_candidates',l_candidate_array)
    #print('right_candidates',r_candidate_array)
    #print('l_lenth',l_len)
    #print('r_length', r_len)
    if i==0:
        l_max =max(l_len)
        r_max =max(r_len)
        for left_var in l_candidate_array:
            slope,intercept,l = left_var
            if l==l_max:
                left_lane.append((slope,intercept))
        for right_var in r_candidate_array:
            slope,intercept,l = right_var
            if l==r_max:
                right_lane.append((slope,intercept))
        left_lane_points = make_coordinate(image, left_lane)
        right_lane_points = make_coordinate(image, right_lane)
        arr = np.array([left_lane_points, right_lane_points])
        print('when i==0')
        print(arr)
    else:
        if not l_candidate_array or not r_candidate_array:
            left_lane_points = all_left_lane_points[i-1]
            right_lane_points = all_right_lane_points[i-1]
            arr = np.array([left_lane_points, right_lane_points])
            print('when left cand or right cand is zero')
            print(arr)
        else:
            l_max =max(l_len)
            r_max =max(r_len)
            for left_var in l_candidate_array:
                slope,intercept,l = left_var
                if l==l_max:
                    left_lane.append((slope,intercept))
            for right_var in r_candidate_array:
                slope,intercept,l = right_var
                if l==r_max:
                    right_lane.append((slope,intercept))
            left_lane_points = make_coordinate(image, left_lane)
            right_lane_points = make_coordinate(image, right_lane)
            if left_lane_points[0] in range(int(all_left_lane_points[i-1][0]-(w*(z/100))),int(all_left_lane_points[i-1][0]+(w*(z/100)))) and left_lane_points[2] in range(int(all_left_lane_points[i-1][2]-(w*(z/100))), int(all_left_lane_points[i-1][2]+(w*(z/100)))):
                left_lane_points = left_lane_points
                print('left normal case')
            else:
                left_lane_points = all_left_lane_points[i-1]
                print('left abnormal case')
            if right_lane_points[0] in range(int(all_right_lane_points[i-1][0]-(w*(z/100))),int(all_right_lane_points[i-1][0]+(w*(z/100)))) and right_lane_points[2] in range(int(all_right_lane_points[i-1][2]-(w*(z/100))), int(all_right_lane_points[i-1][2]+(w*(z/100)))):
                right_lane_points = right_lane_points
                print('right normal case')
            else:
                right_lane_points = all_right_lane_points[i-1]
                print('right abnormal case')
            arr = np.array([left_lane_points, right_lane_points])
            print(arr)

    return arr

def canny(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #st = time.time()
    filtered_image = cv2.bilateralFilter(gray_image,9,75,75)
    #et = time.time()
    #print ('bil time', (et-st)*1000, 'ms')
    #filtered_image = cv2.medianBlur(gray_image,5)
    #ret, thresh1 = cv2.threshold(filtered_image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    canny_image= cv2.Canny(filtered_image,10,30)
    return canny_image

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    #points = np.array([[(0,height), (width/3, height/3), (width* (2/3), height/3) ,(width, height)]])
    points = np.array([[(width/9,height/1.2), (width/4, height/2.75), (width* (3/4), height/2.75) ,(width/1.1, height/1.2)]])
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, np.int32([points]), 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def display_lines(img, lines):
    line_image= np.zeros_like(img)
    for x1, y1, x2, y2 in lines:
        #print(x1,x2,y1,y2)
        cv2.line(line_image,(x1,y1), (x2,y2), (0, 255, 0), 8)
    return line_image

i=0
#t = 0
#total_time = 0
cap = cv2.VideoCapture("D_R_1.mp4")
_, image = cap.read()
h = image.shape[0] 
w = image.shape[1]
out = cv2.VideoWriter('D_R_1_cpf_final_5_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10,(1056,594))
global all_left_lane_points 
global all_right_lane_points
all_left_lane_points = []
all_right_lane_points = []
while(True):
    ret ,img=cap.read()
    st = time.time()
    if ret==True:
        scale_percent = 55# percent of original size
        #print(scale_percent)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        frame = resized_image
        
        canny_image=canny(frame)
        cropped_image = region_of_interest(canny_image)
        d_st = time.time()
        lines= cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=5)
        filtered_lines = filtering(frame, lines)
        all_left_lane_points.append((filtered_lines[0]))
        #print('pre', l_pre_line_points)
        all_right_lane_points.append((filtered_lines[1]))
        line_image = display_lines(frame, filtered_lines)
        combo_image =cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        d_et = time.time()
        detection_time = d_et-d_st
        print('detection time:', detection_time*1000, 'ms')
        et = time.time()
        tt = et-st
        #t = total_time+t
        print('total_time:',tt*1000, 'ms')
        cv2.imshow('result', combo_image)
        cv2.imwrite('DR1_cpf_final_5_'+str(i)+'.jpg',combo_image)
        out.write(combo_image)
        print('frame num:',i)
        i+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()