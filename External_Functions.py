def detect_tf_hardware():
    import tensorflow as tf
    GPU = len(tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: " + str(GPU))
    if GPU > 0:
        my_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_visible_devices(devices= my_devices, device_type='GPU')
        print("Utilising GPU")
    else:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
        print("Could not find a CUDA GPU\nThe programme will use CPU instead")

def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        print("Please just use lowercase y/n... Anything else hurts my CPU")
        reply2 = str(input(' (y/n): ')).lower().strip()
        if reply2[0] == 'y':
            print("Thank you")
            return True
        if reply2[0] == 'n':
            print("No worries")
            return False
        else:
            print("Owww!")
            reply3 = str(input(' (y/n): ')).lower().strip()
            if reply3[0] == 'y':
                print("That hurt, but I forgive you")
                return True
            if reply3[0] == 'n':
                print("My brain is a bit sore now, but we got there in the end!")
                return False
            else:
                print("Why would you do this to me? <{;_;}>")
                return yes_or_no(question)

def dataGenerators ():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range = 0.2,
        zoom_range = 0.2,
        brightness_range=(1.2, 1.5),
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    val_datagen = ImageDataGenerator(
        rescale=1./255
    )
    return train_datagen, test_datagen, val_datagen

def loadImageData (
    train_generator, test_generator, val_generator,
    height, width, train_directory, test_directory,
    validation_directory, batch):
    train_data = train_generator.flow_from_directory(
        train_directory,
        target_size = (height, width),
        class_mode='binary',
        batch_size=batch
    )

    test_data = test_generator.flow_from_directory(
        test_directory,
        target_size = (height, width),
        class_mode='binary',
        batch_size=batch
    )

    val_data = val_generator.flow_from_directory(
        validation_directory,
        target_size = (height, width),
        class_mode='binary',
        batch_size=batch
    )
    return train_data, test_data, val_data

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=12):
        from keras.callbacks import LearningRateScheduler
        import numpy as np
        '''
        Wrapper function to create a LearningRateScheduler with step decay schedule.
        '''
        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch/step_size))
        return LearningRateScheduler(schedule)

def save_model(model_In, save_name = ""):
   # serialize model to JSON
    model_json = model_In.to_json()
    with open("./Saved_Models/{}.json".format(save_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model_In.save_weights("./Saved_Models/{}.h5".format(save_name))
    print("Saved model to disk")

def modelAccuracy(data, model_In, save_name, time):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    true_labels = data.labels
    pred_labels = np.squeeze(np.array(model_In.predict(data) >= 0.5, dtype=np.int))
    cm = confusion_matrix(true_labels, pred_labels)
    data.class_indices

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='mako', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(2) + 0.5, ['Normal', 'Pneumonia'], fontsize=16)
    plt.yticks(np.arange(2) + 0.5, ['Normal', 'Pneumonia'], fontsize=16)
    plt.show()

    results = model_In.evaluate(data, verbose=0)
    accuracy = results[1]
    auc = results[2]
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    Acc_Metrics = [
        "Accuracy: {:.2f}".format(accuracy),
        "AUC: {:.2f}".format(auc),
        "Precision: {:.2f}".format(precision),
        "Recall: {:.2f}".format(recall),
        "Compile Time: {} Seconds". format(time)
    ]
    for metric in Acc_Metrics:
        print(metric)
    with open("./Saved_Models/{}.txt".format(save_name), "w") as Acc_File:
        for metric in Acc_Metrics:
            Acc_File.write(metric + "\n")
    Acc_File.close()

def load_model(model_name = ""):
    from tensorflow.keras.models import model_from_json
    from colorama import Fore, Style
    import h5py
    print(Fore.CYAN + Style.BRIGHT + "Loading Precomputed Model")
    json_file = open("./model/{}.json".format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./model/{}.h5".format(model_name))
    print(Fore.CYAN + Style.BRIGHT + "Loaded model from disk")
    return model