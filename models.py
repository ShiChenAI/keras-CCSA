from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def tiny_base_model(input_shape=(16,16,1)):
    input = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(input)
    x = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(units=120, activation='relu')(x)
    x = Dense(units=84, activation='relu')(x)
    
    return Model(input, x)

def classifier(x):
    x = Dropout(0.5)(x)
    x = Dense(units=10, activation='softmax', name='classification')(x)

    return x