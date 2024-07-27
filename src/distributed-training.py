from __future__ import absolute_import, division, print_function

import argparse
import json
import os

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers, models


def make_datasets_unbatched():
    BUFFER_SIZE = 10000

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    datasets, _ = tfds.load(name='fashion_mnist', with_info=True, as_supervised=True)
    return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)


def build_and_compile_cnn_model():
    print("Training CNN model")
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1), name='image_bytes'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_and_compile_cnn_model_with_batch_norm():
    print("Training CNN model with batch normalization")
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1), name='image_bytes'),
        layers.Conv2D(32, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_and_compile_cnn_model_with_dropout():
    print("Training CNN model with dropout")
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1), name='image_bytes'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def decay(epoch):
    if epoch < 3:
        return 1e-3
    if 3 <= epoch < 7:
        return 1e-4
    return 1e-5


def _preprocess(bytes_inputs):
    decoded = tf.io.decode_jpeg(bytes_inputs, channels=1)
    resized = tf.image.resize(decoded, size=(28, 28))
    return tf.cast(resized, dtype=tf.uint8)


def _get_serve_image_fn(model):
    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string, name='image_bytes')])
    def serve_image_fn(bytes_inputs):
        decoded_images = tf.map_fn(_preprocess, bytes_inputs, dtype=tf.uint8)
        return model(decoded_images)
    return serve_image_fn


def main(args):
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CollectiveCommunication.AUTO))

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    with strategy.scope():
        ds_train = make_datasets_unbatched().batch(BATCH_SIZE).repeat()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds_train = ds_train.with_options(options)

        if args.model_type == "cnn":
            multi_worker_model = build_and_compile_cnn_model()
        elif args.model_type == "dropout":
            multi_worker_model = build_and_compile_cnn_model_with_dropout()
        elif args.model_type == "batch_norm":
            multi_worker_model = build_and_compile_cnn_model_with_batch_norm()
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

    checkpoint_dir = args.checkpoint_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f'\nLearning rate for epoch {epoch + 1} is {multi_worker_model.optimizer.lr.numpy()}')

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]

    multi_worker_model.fit(ds_train, epochs=10, steps_per_epoch=70, callbacks=callbacks)

    def is_chief():
        return TASK_INDEX == 0

    model_path = args.saved_model_dir if is_chief() else f"{args.saved_model_dir}/worker_tmp_{TASK_INDEX}"

    multi_worker_model.save(model_path)

    signatures = {
        "serving_default": _get_serve_image_fn(multi_worker_model).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='image_bytes')
        )
    }

    tf.saved_model.save(multi_worker_model, model_path, signatures=signatures)


if __name__ == '__main__':
    os.environ['NCCL_DEBUG'] = 'INFO'
    tfds.disable_progress_bar()

    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    TASK_INDEX = tf_config.get('task', {}).get('index', 0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_dir', type=str, required=True, help='Tensorflow export directory.')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Tensorflow checkpoint directory.')
    parser.add_argument('--model_type', type=str, required=True, help='Type of model to train.')

    parsed_args = parser.parse_args()
    main(parsed_args)