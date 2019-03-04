from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, GaussianNoise, Activation, TimeDistributed
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, Reshape, Lambda
from tensorflow.keras.applications.nasnet import NASNetLarge

class MyModel():
    
    def __init__(self):
        
        self.base_model = NASNetLarge(include_top=False, weights='imagenet', input_shape=(299 ,299 ,3))
        self.model = self.get_model()
        
    def get_model(self):
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)

        predictions = Dense(6, activation='softmax')(x)

        model = Model(inputs=self.base_model.input, outputs=predictions)

        return model
    