import argparse
import os
from tensorflow import keras
import tensorflow as tf
from pprint import pprint
import time
import logging
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from data_process.data_loader import data_loader
from utils.metrics import micro_f1, macro_f1
from utils.params_utils import get_params


def TextCNN(max_sequence_length, max_token_num, embedding_dim, output_dim, model_img_path=None, embedding_matrix=None):
    """
    TextCNN:
    1. embedding layers,
    2.convolution layer,
    3.max-pooling,
    4.softmax layer.
    """
    x_input = Input(shape=(max_sequence_length,))
    logging.info("x_input.shape: %s" % str(x_input.shape))  # (?, 60)

    if embedding_matrix is None:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length)(x_input)
    else:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length,
                          weights=[embedding_matrix], trainable=True)(x_input)

    logging.info("x_emb.shape: %s" % str(x_emb.shape))  # (?, 60, 300)

    pool_output = []
    kernel_sizes = [2, 3, 4]
    for kernel_size in kernel_sizes:
        c = Conv1D(filters=2, kernel_size=kernel_size, strides=1)(x_emb)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
        logging.info("kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(c.shape), str(p.shape)))

    pool_output = concatenate([p for p in pool_output])
    logging.info("pool_output.shape: %s" % str(pool_output.shape))  # (?, 1, 6)

    x_flatten = Flatten()(pool_output)  # (?, 6)
    y = Dense(output_dim, activation='sigmoid')(x_flatten)  # (?, 2)

    logging.info("y.shape: %s \n" % str(y.shape))

    model = Model([x_input], outputs=[y])

    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    model.summary()

    return model



def train(X_train, X_test, y_train, y_test, params, save_path):
    print("\nTrain...")
    model = build_model(params)

    early_stopping = EarlyStopping(monitor='val_micro_f1', patience=10, mode='max')

    history = model.fit(X_train, y_train,
                        batch_size=params.batch_size,
                        epochs=params.epochs,
                        workers=params.workers,
                        use_multiprocessing=True,
                        callbacks=[early_stopping],
                        validation_data=(X_test, y_test))

    print("\nSaving model...")
    keras.models.save_model(model, save_path)
    pprint(history.history)


def build_model(params):
    model = TextCNN(max_sequence_length=params.padding_size, max_token_num=params.vocab_size,
                    embedding_dim=params.embed_size,
                    output_dim=params.num_classes)
    model.compile(tf.optimizers.Adam(learning_rate=params.learning_rate),
                  loss='binary_crossentropy',
                  metrics=[micro_f1, macro_f1])

    model.summary()
    return model


if __name__ == '__main__':
    params = get_params()
    print('Parameters:', params, '\n')

    if not os.path.exists(params.results_dir):
        os.mkdir(params.results_dir)
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    os.mkdir(os.path.join(params.results_dir, timestamp))
    os.mkdir(os.path.join(params.results_dir, timestamp, 'log/'))

    X_train, X_test, y_train, y_test = data_loader(params)

    train(X_train, X_test, y_train, y_test, params, os.path.join(params.results_dir, 'TextCNN.h5'))
