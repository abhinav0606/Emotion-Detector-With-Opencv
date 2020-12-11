from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
def retun_model():
    train="train"
    test="test"
    # data_generator
    train_data_generator=ImageDataGenerator(1./255)
    test_data_generator=ImageDataGenerator(1./255)
    # actuall generator
    train_generator=train_data_generator.flow_from_directory(
        train,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical"
    )
    test_generator=train_data_generator.flow_from_directory(
        test,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical"
    )
    # creating the model
    emotion=Sequential()
    emotion.add(Conv2D(32,input_shape=(48,48,1),kernel_size=(2,2),activation="relu"))
    emotion.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
    emotion.add(MaxPooling2D(pool_size=(2,2)))
    emotion.add(Dropout(0.25))
    emotion.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
    emotion.add(MaxPooling2D(pool_size=(2,2)))
    emotion.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
    emotion.add(MaxPooling2D(pool_size=(2,2)))
    emotion.add(Dropout(0.25))
    emotion.add(Flatten())
    emotion.add(Dense(1024,activation="relu"))
    emotion.add(Dropout(0.5))
    emotion.add(Dense(7,activation="softmax"))

    return emotion