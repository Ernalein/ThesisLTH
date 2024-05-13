# imports
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from scipy.io import loadmat


def load_and_prep_cifar(batch_size, shuffle_size):
    # load data set
    (train_ds, test_ds), ds_info = tfds.load(name="cifar10", split=["train","test"], as_supervised=True, with_info=True)
    # tfds.show_examples(train_ds, ds_info)
    
    def prepare_cifar10_data(ds):
        #convert data from uint8 to float32
        ds = ds.map(lambda img, target: (tf.cast(img, tf.float32), target))
        #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
        ds = ds.map(lambda img, target: ((img/128.)-1., target))
        #create one-hot targets
        ds = ds.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
        #cache this progress in memory, as there is no need to redo it; it is deterministic after all
        ds = ds.cache()
        #shuffle, batch, prefetch
        ds = ds.shuffle(shuffle_size).batch(batch_size).prefetch(2)
        #return preprocessed dataset
        return ds
    
    # prepare data
    train_dataset = train_ds.apply(prepare_cifar10_data)
    test_dataset = test_ds.apply(prepare_cifar10_data)
    
    return train_dataset, test_dataset

def load_and_prep_svhn(batch_size, shuffle_size):

    # copied from https://github.com/Holleri/code_bachelor_thesis/blob/main/code_LTH/LTH_IMP_self_implementation_conv2_SVHN.ipynb

    num_classes = 10
    img_rows, img_cols = 32, 32

    train = loadmat("data/SVHN/train_32x32.mat")
    test = loadmat("data/SVHN/test_32x32.mat")

    X_train = np.array(train["X"])
    y_train = np.array(train["y"])

    X_test = np.array(test["X"])
    y_test = np.array(test["y"])

    # bring into right format (shape = (nr_images, height, width, channels))
    X_train = np.swapaxes(X_train, 3, 0)
    X_train = np.swapaxes(X_train, 3, 1)
    X_train = np.swapaxes(X_train, 3, 2)

    X_test = np.swapaxes(X_test, 3, 0)
    X_test = np.swapaxes(X_test, 3, 1)
    X_test = np.swapaxes(X_test, 3, 2)

    # change label '10' to '0'
    y_train = np.array([x if x != [10] else [0] for x in y_train])
    y_test = np.array([x if x != [10] else [0] for x in y_test])

    # Convert datasets to floating point types-
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize the training and testing datasets-
    X_train /= 255.0
    X_test /= 255.0

    # convert class vectors/target to binary class matrices or one-hot encoded values-
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # split training set into train and validation (10%)
    l = len(X_train)
    X_val = X_train[:int(l/10)]
    X_train = X_train[int(l/10):]

    y_val = y_train[:int(l/10)]
    y_train = y_train[int(l/10):]

    # combine into real datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.batch(batch_size = batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size = batch_size)
    
    return train_ds, val_ds


# downloading and preprocessing CIINIC-10 dataset

def load_and_prep_cinic(batch_size):

    # copied from https://github.com/Holleri/code_bachelor_thesis/blob/main/code_LTH/LTH_IMP_self_implementation_conv2_CINIC10.ipynb

    num_classes = 10
    img_rows, img_cols = 32, 32

    FOLDER_TRAIN_DATASET_CINIC = "data/CINIC-10/train"

    # training set (90% of images)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        FOLDER_TRAIN_DATASET_CINIC, 
        validation_split = 0.1,
        subset="training",
        seed=12,
        image_size=(img_rows, img_cols),
        batch_size = batch_size)

    # validation set (10%)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        FOLDER_TRAIN_DATASET_CINIC,
        validation_split=0.1,
        subset="validation",
        seed=12,
        image_size=(img_rows, img_cols),
        batch_size=batch_size)

    # normalize images
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def load_and_prep_dataset(dataset, batch_size=60, shuffle_size=512):

    train_dataset = None
    test_dataset = None

    if dataset == "CIFAR":
        train_dataset, test_dataset = load_and_prep_cifar(batch_size=batch_size, shuffle_size=shuffle_size)
    elif dataset == "CINIC":
        train_dataset, test_dataset = load_and_prep_cinic(batch_size=batch_size)
    elif dataset == "SVHN":
        train_dataset, test_dataset = load_and_prep_svhn(batch_size=batch_size, shuffle_size=shuffle_size)

    return train_dataset, test_dataset