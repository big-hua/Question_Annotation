import argparse
import os
from tensorflow import keras
import tensorflow as tf
from pprint import pprint
import time

from tensorflow.keras.callbacks import EarlyStopping

from data_process.data_loader import data_loader
from model.text_cnn import TextCNN
from utils.metrics import micro_f1, macro_f1
from utils.params_utils import get_params


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
    if params.model == 'cnn':
        model = TextCNN(max_sequence_length=params.padding_size, max_token_num=params.vocab_size,
                        embedding_dim=params.embed_size,
                        output_dim=params.num_classes)
        model.compile(tf.optimizers.Adam(learning_rate=params.learning_rate),
                      loss='binary_crossentropy',
                      metrics=[micro_f1, macro_f1])

    else:

        pass

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
