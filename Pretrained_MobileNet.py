import tensorflow as tf
from External_Functions import dataGenerators, detect_tf_hardware, loadImageData, modelAccuracy, save_model, step_decay_schedule

detect_tf_hardware()
train_dir = './chest_xray/train'
test_dir = './chest_xray/test'
# Validation directory holds sample files to be used purely for model validation split
val_dir = './/chest_xray/val'

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

train_datagen, test_datagen, val_datagen = dataGenerators()
train_data, test_data, val_data = loadImageData(
    train_generator=train_datagen, test_generator=test_datagen, val_generator=val_datagen,
    height=IMG_HEIGHT, width=IMG_WIDTH, train_directory=train_dir,
    test_directory=test_dir, validation_directory=val_dir, batch=BATCH_SIZE)

mobilenet = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg',
)
mobilenet.trainable = False
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
pretrained_model = mobilenet(inputs, training=False)
dense = tf.keras.layers.Dense(1024, activation='relu')(pretrained_model)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
model = tf.keras.Model(inputs, outputs)
print(model.summary())

EPOCHS = 50
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)
lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=3)
history = model.fit(
    train_data,
    validation_data=val_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        lr_sched
    ]
)
save_model(model_In=model, save_name="Pretrained_ImageNet")
modelAccuracy(data = test_data, model_In=model)