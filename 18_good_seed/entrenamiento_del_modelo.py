import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def load_data(path, subset='training'):
    labels = pd.read_csv(path + 'labels.csv')
    
    data_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1/255
    )
    
    data_gen_flow = data_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset=subset,
        seed=12345)

    return data_gen_flow

def create_model(input_shape):
    
    backbone = ResNet50(weights='imagenet', 
                        input_shape=input_shape,
                        include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=3):

    model.fit(train_data, validation_data=test_data, batch_size=batch_size, epochs=epochs, verbose=1)

    return model
    
input_shape = (224, 224, 3)

train = load_data('datasets/faces/')
test = load_data('datasets/faces/', subset='validation')
model = create_model(input_shape)
model = train_model(model, train, test)