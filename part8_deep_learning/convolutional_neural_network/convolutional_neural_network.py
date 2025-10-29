import tensorflow as tf

# Paths
TRAIN_DIR = "part8_deep_learning/convolutional_neural_network/dataset/training_set"
TEST_DIR  = "part8_deep_learning/convolutional_neural_network/dataset/test_set"
SINGLE_IMG = "part8_deep_learning/convolutional_neural_network/dataset/single_prediction/cat_or_dog_3.jpg"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 1337

# Part 1 - Data Preprocessing
train_set = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

test_set = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,  # keep evaluation deterministic
)

# Save class names
class_names = train_set.class_names
print("class_names:", class_names)   # check order (alphabetical by folder names)

# inspect a batch of labels
for imgs, labels in train_set.take(1):
    print("sample labels (0/1):", labels[:10].numpy().reshape(-1))
    break  # just one batch

# Define an augmentation pipeline
augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)

# Rescale to match `rescale=1./255` behavior from ImageDataGenerator
# Rescale and apply augmentation to training set only
train_set = train_set.map(lambda x, y: (x / 255.0, y))
train_set = train_set.map(lambda x, y: (augmenter(x, training=True), y))

# Test set: rescale only (no augmentation)
test_set  = test_set.map(lambda x, y: (x / 255.0, y))

# # Performance tweaks
AUTOTUNE = tf.data.AUTOTUNE
train_set = train_set.cache().prefetch(AUTOTUNE)
test_set  = test_set.cache().prefetch(AUTOTUNE)

# Part 2 - Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
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
cnn.fit(x = train_set, validation_data = test_set, epochs = 25)

# -----------------------
# Single-image prediction (same flow: load -> array -> expand_dims)
# -----------------------
test_image = tf.keras.utils.load_img(SINGLE_IMG, target_size=IMG_SIZE)
test_image = tf.keras.utils.img_to_array(test_image)
test_image = test_image / 255.0                     # match dataset scaling
test_image = tf.expand_dims(test_image, axis=0)     # (1, 64, 64, 3)


result = cnn.predict(test_image)
if result[0][0] == 1: # the first dimention is the batch and the second dimention is the single image in that batch
  prediction = 'dog'
else:
  prediction = 'cat'
print(prediction)