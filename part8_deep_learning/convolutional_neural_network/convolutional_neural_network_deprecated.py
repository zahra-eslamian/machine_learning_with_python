# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Part 1 - Data Preprocessing
# In CNN, we do some data augmentation (like rotation, zoom in/out, ...) to prevent overfiting
# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255, # this line will apply feature scaling to every      single pixel by dividing their value by 255. This will give us a value in [0:1] for each pixel. As said before, feature scaling is absolutely necessray for NNs.
                                   shear_range = 0.2, #the following 3 lines are the transformations we apply on the images 
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

#next, we connect that obj from ImageDataGenerator class (train_datagen) to the images of our training set
training_set = train_datagen.flow_from_directory('part8_deep_learning/convolutional_neural_network/dataset/training_set',
                                                 target_size = (64, 64), #final size of the image when they are fed into the CNN
                                                 batch_size = 32,
                                                 class_mode = 'binary') #he type of classification: binary or multi-class. Here, we have binary (Cat or Dog)

# Preprocessing the Test set
# for the test set, of course we don't want to apply any transformation but we keep them intact
# we just do the same feature scaling as we did with our training set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('part8_deep_learning/convolutional_neural_network/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
# in Conv2D class, filters: the number of output filters/kernels in the convolution
# kernel_size: the size of the kernel (3: 3*3)
# input_shape: when building the first layer of our network, whether is a convolution layer or a dense layer, we should specify the shape of the input data. For images (2D data): (height,width,channels); "height": number of pixels vertically, "width": number of pixels horizontally, "channels": number of color channels (1 = grayscale, 3 = RGB, 4 = RGBA, etc.)
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Part 4 - Making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('part8_deep_learning/convolutional_neural_network/dataset/single_prediction/cat_or_dog.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)

# Neural networks in TensorFlow/Keras expect input data in the form (batch_size,height,width,channels)
# But when you load a single image (with image.load_img and img_to_array), its shape is usually:
# (height,width,channels)
# The following line adds the batch dimension to your image, turning a single image into a "batch of one image," so the CNN can process it.
test_image = np.expand_dims(test_image, axis = 0) 
result = cnn.predict(test_image)
training_set.class_indices # to understand which number corresponds to which class. here, 1: Dog, 0: Cat
if result[0][0] == 1: # the first dimention is the batch and the second dimention is the single image in that batch
  prediction = 'dog'
else:
  prediction = 'cat'
print(prediction)