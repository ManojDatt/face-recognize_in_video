# # # import cv2
# # # import numpy as np
# # # from matplotlib import pyplot as plt

# # # def process_img(img_rgb, template, count):
# # #     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# # #     w, h = template.shape[::-1]

# # #     res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# # #     threshold = 0.8
# # #     loc = np.where( res >= threshold)
# # #     for pt in zip(*loc[::-1]):
# # #         cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

# # #     # This will write different res.png for each frame. Change this as you require
# # #     # cv2.imwrite('res{0}.png'.format(count),img_rgb)   


# # # def main():
# # #     vidcap = cv2.VideoCapture('Panasonic_HDC_TM_700_P_50i.avi')
# # #     template = cv2.imread('test.png',0)  # open template only once
# # #     count = 0
# # #     while True:
# # #       success,image = vidcap.read()
# # #       if not success: break         # loop and a half construct is useful
# # #       print ('Read a new frame: ', success)
# # #       process_img(image, template, count)
# # #       count += 1

# # # if __name__ == '__main__':
# # #   main()

# # #!/usr/bin/env python

# # # Python 2/3 compatibility
# # from __future__ import print_function
# # # Allows use of print like a function in Python 2.x

# # # Import OpenCV and Numpy modules
# # import numpy as np
# # import cv2
 
# # try:
# #     # Create a named window to display video output
# #     cv2.namedWindow('Watermark', cv2.WINDOW_NORMAL)
# #     # This section is the same from previous Image example.
# #     # Load logo image
# #     dog = cv2.imread('face.jpg')
# #     # 
# #     rows,cols,channels = dog.shape
# #     # Convert the logo to grayscale
# #     dog_gray = cv2.cvtColor(dog,cv2.COLOR_BGR2GRAY)
# #     # Create a mask of the logo and its inverse mask
# #     ret, mask = cv2.threshold(dog_gray, 1, 255, cv2.THRESH_BINARY)
# #     mask_inv = cv2.bitwise_not(mask)
# #     # Now just extract the logo
# #     dog_fg = cv2.bitwise_and(dog,dog,mask = mask)
    
# #     # Initialize Default Video Web Camera for capture.
# #     webcam = cv2.VideoCapture("Panasonic_HDC_TM_700_P_50i.avi")
# #     # Check if Camera initialized correctly
# #     success = webcam.isOpened()
# #     # if success == False:
# #     #     print('Error: Camera could not be opened')
# #     # else:
# #     #     print('Sucess: Grabbing the camera')
# #     #     webcam.set(cv2.CAP_PROP_FPS,30);
# #     #     webcam.set(cv2.CAP_PROP_FRAME_WIDTH,1024);
# #     #     webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,768);

# #     while(True):
# #         # Read each frame in video stream
# #         ret, frame = webcam.read()
# #         # Perform operations on the video frames here
# #         # To put logo on top-left corner, create a Region of Interest (ROI)
# #         roi = frame[0:rows, 0:cols ] 
# #         # Now blackout the area of logo in ROI
# #         frm_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# #         # Next add the logo to each video frame
# #         dst = cv2.add(frm_bg,dog_fg)
# #         frame[0:rows, 0:cols ] = dst
# #         # Overlay Text on the video frame with Exit instructions
# #         font = cv2.FONT_HERSHEY_SIMPLEX
# #         # cv2.putText(frame, "Type q to Quit:",(50,700), font, 1,(255,255,255),2,cv2.LINE_AA)
# #         # Display the resulting frame
# #         cv2.imshow('Watermark',frame)
# #         # Wait for exit key "q" to quit
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             print('Quitting ...')
# #             break

# #     # Release all resources used
# #     webcam.release()
# #     cv2.destroyAllWindows()

# # except cv2.error as e:
# #     print('Please correct OpenCV Error')


# import numpy as np
# import cv2

# cap = cv2.VideoCapture('slow.flv')
# print(dir(cap))
# # take first frame of the video
# ret,frame = cap.read()
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# print(frame_height, frame_width)
# # setup initial location of window
# r,h,c,w = 250,90,400,125  # simply hardcoded the values
# track_window = (c,r,w,h)

# # set up the ROI for tracking
# roi = frame[r:r+h, c:c+w]
# hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
# term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

# while(1):
#     ret ,frame = cap.read()

#     if ret == True:
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

#         # apply meanshift to get the new location
#         ret, track_window = cv2.meanShift(dst, track_window, term_crit)

#         # Draw it on image
#         x,y,w,h = track_window
#         img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
#         cv2.imshow('img2',frame)

#         k = cv2.waitKey(60) & 0xff
#         if k == 27:
#           break
#         else:
#           cv2.imwrite(chr(k)+".jpg",img2)

#     else:
#         break

# cv2.destroyAllWindows()
# cap.release()

# import numpy
# import cv2

# cap = cv2.VideoCapture('slow.flv')

# fourcc = cv2.cv.FOURCC(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# while(1):

#     # read the frames
#     _,frame = cap.read()
#         #A line
#     cv2.line(frame, (500, 400), (640, 480),(0,255,0), 3)


#     cv2.putText(frame, "test!",(105, 105),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(225,0,0))
#     out.write(frame)
#     #if key pressed is 'Esc', exit the loop
#     cv2.imshow('frame',frame)

#     if cv2.waitKey(33)== 27:
#         break
# out.release()

# cv2.destroyAllWindows()

#! /usr/bin/python

# import cv2
# from itertools import count

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# # Load the overlay image: glasses.png
# imgGlasses = cv2.imread('eye.jpeg')

# #Check if the files opened
# if  imgGlasses is None :
#     exit("Could not open the image")
# if  face_cascade.empty() :
#     exit("Missing: haarcascade_frontalface_default.xml")
# if  eye_cascade.empty() :
#     exit("Missing: haarcascade_eye.xml")


# # Create the mask for the glasses
# imgGlassesGray = cv2.cvtColor(imgGlasses, cv2.COLOR_BGR2GRAY)
# ret, orig_mask = cv2.threshold(imgGlassesGray, 10, 255, cv2.THRESH_BINARY)

# #orig_mask = imgGlasses[:,:,3]

# # Create the inverted mask for the glasses
# orig_mask_inv = cv2.bitwise_not(orig_mask)

# # Convert glasses image to BGR
# # and save the original image size (used later when re-sizing the image)
# imgGlasses = imgGlasses[:,:,0:3]
# origGlassesHeight, origGlassesWidth = imgGlasses.shape[:2]

# #cv2.imshow('Video', imgGlasses)
# #cv2.waitKey()


# video_capture = cv2.VideoCapture("Desi_Da_Recard_-_Ninja.mp4")
# fourcc = cv2.cv.FOURCC(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# if not video_capture.isOpened() :
#     exit('The Camera is not opened')


# counter = count(1)

# while True:
#     print "Iteration %d" % counter.next()
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)

#         #cv2.imshow('Video', roi_gray)
#         #cv2.waitKey()

#         #print 'X:%i, Y:%i, W:%i, H:%i' % (x, y, w, h)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#             print 'EX:%i, EY:%i, EW:%i, EH:%i' % (ex, ey, ew, eh)

#         for (ex, ey, ew, eh) in eyes:
#             glassesWidth = 3*ew
#             glassesHeight = glassesWidth * origGlassesHeight / origGlassesWidth

#             # Center the glasses on the bottom of the nose
#             x1 = ex - (glassesWidth/4)
#             x2 = ex + ew + (glassesWidth/4)
#             y1 = ey + eh - (glassesHeight/2)
#             y2 = ey + eh + (glassesHeight/2)

#                 # Check for clipping
#             if x1 < 0:
#                 x1 = 0
#             if y1 < 0:
#                 y1 = 0
#             if x2 > w:
#                 x2 = w
#             if y2 > h:
#                 y2 = h

#             # Re-calculate the width and height of the glasses image
#             glassesWidth = x2 - x1
#             glassesHeight = y2 - y1

#             # Re-size the original image and the masks to the glasses sizes
#             # calcualted above
#             glasses = cv2.resize(imgGlasses, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)
#             mask = cv2.resize(orig_mask, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)
#             mask_inv = cv2.resize(orig_mask_inv, (glassesWidth,glassesHeight), interpolation = cv2.INTER_AREA)

#             # take ROI for glasses from background equal to size of glasses image
#             roi = roi_color[y1:y2, x1:x2]

#             # roi_bg contains the original image only where the glasses is not
#             # in the region that is the size of the glasses.
#             roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

#             # roi_fg contains the image of the glasses only where the glasses is
#             roi_fg = cv2.bitwise_and(glasses,glasses,mask = mask)

#             # join the roi_bg and roi_fg
#             dst = cv2.add(roi_bg,roi_fg)

#             # place the joined image, saved to dst back over the original image
#             roi_color[y1:y2, x1:x2] = dst
#     #break
#     #Display the resulting frame
#     # cv2.imshow('Video', frame)
#     out.write(frame)
#     cv2.waitKey()   
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()





# /*
import face_recognition
import cv2

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture('Desi_Da_Recard_-_Ninja.avi')
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fourcc = cv2.cv.FOURCC(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width,frame_height))
# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("known.png")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.png")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Actor",
    "Actories"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    out.write(frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
out.release()
cv2.destroyAllWindows()
# */