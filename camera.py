from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
# directory
train="train"
test="test"
# data generator
test_data_generator=ImageDataGenerator(1./255)
train_data_generator=ImageDataGenerator(1./255)
# creating the generator
test_generator=test_data_generator.flow_from_directory(
    test,target_size=(48,48),batch_size=64,color_mode="grayscale",
    class_mode="categorical"
)
train_generator=train_data_generator.flow_from_directory(
    train,target_size=(48,48),batch_size=64,color_mode="grayscale",
    class_mode="categorical"
)

emotion=Sequential()
emotion.add(Conv2D(32,input_shape=(48,48,1),kernel_size=(3,3),activation="relu"))
emotion.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
emotion.add(MaxPooling2D(pool_size=(2,2)))
emotion.add(Dropout(0.25))
emotion.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
emotion.add(MaxPooling2D(pool_size=(2,2)))
emotion.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
emotion.add(MaxPooling2D(pool_size=(3,3)))
emotion.add(Dropout(0.25))
emotion.add(Flatten())
emotion.add(Dense(1024,activation="relu"))
emotion.add(Dropout(0.5))
emotion.add(Dense(7,activation="softmax"))
emotion.load_weights("model1.h5")
import cv2
import numpy as np
capture=cv2.VideoCapture(0)
faces=cv2.CascadeClassifier("frontal_face.xml")
cv2.ocl.setUseOpenCL(False)
Emotion_dictionary={
    0:"Angry",
    1:"Disgusted",
    2:"Fearfull",
    3:"Happy",
    4:"Neutral",
    5:"Sad",
    6:"Surprised"
}
while True:
    ret,frame=capture.read()
    frame=cv2.flip(frame,1)
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=faces.detectMultiScale(gray_frame,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),[0,255,0],2)
        cropped_gray_image=gray_frame[y:y+h,x:x+w]
        new_size_frame=np.expand_dims(np.expand_dims(cv2.resize(cropped_gray_image,(48,48)),-1),0)
        prediction=emotion.predict(new_size_frame)
        index=int(np.argmax(prediction))
        cv2.putText(frame, Emotion_dictionary[index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.imshow("Abhinav's Frame",frame)
    if cv2.waitKey(1)==13:
        break
capture.release()
cv2.destroyAllWindows()
