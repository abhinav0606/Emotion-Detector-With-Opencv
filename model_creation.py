from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

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

emotion.compile(loss="categorical_crossentropy",optimizer=(Adam(lr=0.0001,decay=1e-6)),metrics=['accuracy'])
info=emotion.fit_generator(
    train_generator,
    steps_per_epoch=28709//64,
    epochs=50,
    validation_data=test_generator,
    validation_steps=7178//64
)
emotion.save_weights("model1.h5")