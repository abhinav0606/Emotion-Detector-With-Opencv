import cv2
import numpy as np
from model_return import retun_model
model=retun_model().load_weights("model1.h5")
capture=cv2.VideoCapture(0)
faces=cv2.CascadeClassifier("frontal_face.xml")
cv2.ocl.setUseOpenCL(False)
while True:
    ret,frame=capture.read()
    frame=cv2.flip(frame,1)
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=faces.detectMultiScale(gray_frame,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),[0,255,0],2)
        cropped_gray_image=gray_frame[y:y+h,x:x+w]
        new_size_frame=np.expand_dims(np.expand_dims(cv2.resize(cropped_gray_image,(48,48)),-1),0)
        cv2.imshow("Abhinav's Frame",cropped_gray_image)
    if cv2.waitKey(1)==13:
        break
capture.release()
cv2.destroyAllWindows()
