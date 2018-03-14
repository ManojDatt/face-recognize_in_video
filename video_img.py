#!/usr/bin/env python
# http://sublimerobots.com/2015/02/dancing-mustaches/
import Image
import face_recognition #https://github.com/ageitgey/face_recognition
import cv2  # OpenCV Library
import os
import random
import numpy as np
import time
start = time.time()

def transparentOverlay(src , overlay , pos=(0,0, 0, 0),scale = 1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    overlay = cv2.resize(overlay,(pos[2],pos[3]))
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    

    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f avi '{output}.avi'".format(input = avi_file_path, output = output_name))
    return True 
#-----------------------------------------------------------------------------
#       Load and configure Haar Cascade Classifiers
#-----------------------------------------------------------------------------
 
# location of OpenCV Haar Cascade Classifiers:
 
# xml files describing our haar cascade classifiers
faceCascadeFilePath = "haarcascade_frontalface_default.xml"
noseCascadeFilePath = "haarcascade_mcs_nose.xml"
mouthCascadeFilePath = "haarcascade_mcs_mouth.xml"
eyeCascadeFilePath = "haarcascade_eye.xml"
 
# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
mouthCascade = cv2.CascadeClassifier(mouthCascadeFilePath)
eyeCascade = cv2.CascadeClassifier(eyeCascadeFilePath)

known_image = face_recognition.load_image_file("known.png")
biden_encoding = face_recognition.face_encodings(known_image)[0]


#-----------------------------------------------------------------------------
#       Load and configure mustache (.png with alpha transparency)
#-----------------------------------------------------------------------------
# im = Image.open('face.jpg')
# im.save('face.png')

# Load our overlay image: mustache.png
imgNose = cv2.imread('nose.png',cv2.IMREAD_UNCHANGED)
imgMouth = cv2.imread('mouth_smile.png', cv2.IMREAD_UNCHANGED) 
overlayImage = cv2.imread("shahrukh_khan.png" , cv2.IMREAD_UNCHANGED)

#-------------------------------------------------
video_capture = cv2.VideoCapture('Desi_Da_Recard_-_Ninja.avi')
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fourcc = cv2.cv.FOURCC(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width,frame_height))

while True:
    # Capture video feed
    ret, frame = video_capture.read()
 
    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Detect faces in input video stream
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
 
   # Iterate over each face found
    for (x, y, w, h) in faces:
        # Un-comment the next line for debug (draw box around all faces)
        # face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        nose = noseCascade.detectMultiScale(roi_gray)
        mouth = mouthCascade.detectMultiScale(roi_gray)
        eye = eyeCascade.detectMultiScale(roi_gray)

        try:
            
            # Load a second sample picture and learn how to recognize it.
            cv2.imwrite('unknown.png',roi_color)
            biden_image = face_recognition.load_image_file("unknown.png")
            unknown_encoding = face_recognition.face_encodings(biden_image)[0]

            results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
            if True in results:
                transparentOverlay(frame,overlayImage,(x, y, w, h+2),0.7)
                # for (nx,ny,nw,nh) in nose:
                #     transparentOverlay(roi_color,imgNose,(nx, ny, nw, nh),0.7)
                #     # cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)

                # for (nx,ny,nw,nh) in mouth:
                #     transparentOverlay(roi_color,imgMouth,(nx, ny, nw, nh),0.7)
                #     # Un-comment the next line for debug (draw box around the nose)
                    # cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
        except:
            pass
 
    # Display the resulting frame
    cv2.imshow('Video', frame)
    out.write(frame)
    # press any key to exit
    # NOTE;  x86 systems may need to remove: " 0xFF == ord('q')"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything is done, release the capture
video_capture.release()
out.release()
cv2.destroyAllWindows()
done = time.time()
elapsed = done - start
print("Elapsed: {}".format(elapsed))
