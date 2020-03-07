'''
    Treinamento de rede neural convolucional para classificar fotos de jovens 
    e idosos, utilizando a base 'young-old dataset'
'''

# Importações

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing import image

# Criação da rede neural

classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Carregamento das bases de treinamento e de teste

generator_train = ImageDataGenerator(rescale = 1./255)
generator_test = ImageDataGenerator(rescale = 1./255)

base_train = generator_train.flow_from_directory('dataset\\train',
                                                 target_size = (64, 64), 
                                                 batch_size = 20,
                                                 class_mode = 'binary')
base_test = generator_test.flow_from_directory('dataset\\test',
                                               target_size = (64, 64),
                                               batch_size = 20,
                                               class_mode = 'binary')

# Treinamento

classifier.fit_generator(base_train, steps_per_epoch = 482/20,
                         epochs = 50, validation_data = base_test,
                         validation_steps = 154/20)

# Avaliação e previsões

img_test = image.load_img('dataset\\test\\young\\2.cde5ca66cdd6c9345b7f854f630571e4.jpg',
                              target_size = (64,64))
img_test = image.img_to_array(img_test)
img_test /= 255
img_test = np.expand_dims(img_test, axis = 0)
predict = classifier.predict(img_test)
predict = (predict > 0.5)

#base_train.class_indices

if predict == False:
    print('Idoso')
else:
    print('Jovem')

# Salvar

'''classifier_json = classifier.to_json()
with open('classifier_young-old1.json', 'w') as json_file:
    json_file.write(classifier_json)
classifier.save_weights('classifier_young-old1.h5')'''

# Carregar

'''
a = open('classifier_young-old1.json', 'r')
network_structure = a.read()
a.close()
classifier = model_from_json(network_structure)
classifier.load_weights('classifier_young-old1.h5')
'''
