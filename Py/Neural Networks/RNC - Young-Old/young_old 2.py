'''
    Treinamento de rede neural convolucional para classificar fotos de jovens 
    e idosos, utilizando a base 'dataset young-old'
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
classifier.add(Conv2D(32, (3,3), input_shape = (100, 100, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32, (3,3), input_shape = (100, 100, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 100, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 100, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Carregamento das bases de treinamento e de teste

generator_train = ImageDataGenerator(rescale = 1./255, rotation_range = 7,
                                     horizontal_flip = True, shear_range = 0.2,
                                     height_shift_range = 0.07, 
                                     zoom_range = 0.2)
generator_test = ImageDataGenerator(rescale = 1./255)

base_train = generator_train.flow_from_directory('dataset old_young\\train',
                                                 target_size = (100, 100), 
                                                 batch_size = 10,
                                                 class_mode = 'binary')
base_test = generator_test.flow_from_directory('dataset old_young\\test',
                                               target_size = (100, 100),
                                               batch_size = 10,
                                               class_mode = 'binary')

# Treinamento

classifier.fit_generator(base_train, steps_per_epoch = 386/10,
                         epochs = 50, validation_data = base_test,
                         validation_steps = 129/10)

# Avaliação e previsões

img_test = image.load_img('dataset old_young\\test\\young\\1.3CB079C700000578-4175642-image-a-24_1485865098003.jpg',
                              target_size = (100,100))
img_test = image.img_to_array(img_test)
img_test /= 255
img_test = np.expand_dims(img_test, axis = 0)
predict = classifier.predict(img_test)
predict_c = (predict > 0.5)

#base_train.class_indices

print(f"Previsão: {predict}")

if predict_c == False:
    print('Imagem classificada como: Idoso')
else:
    print('Imagem classificada como: Jovem')

# Salvar

'''classifier_json = classifier.to_json()
with open('classifier_young-old2.json', 'w') as json_file:
    json_file.write(classifier_json)
classifier.save_weights('classifier_young-old2.h5')'''

# Carregar

'''
a = open('classifier_young-old2.json', 'r')
network_structure = a.read()
a.close()
classifier = model_from_json(network_structure)
classifier.load_weights('classifier_young-old2.h5')
'''

