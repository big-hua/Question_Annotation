import argparse

from utils.config import data_path, vocab_save_dir, result_path



def get_params():
    # 获得参数
    parser = argparse.ArgumentParser(description='This is the TextCNN test project.')
    parser.add_argument('--data_path', default=data_path, type=str, help='data path')
    parser.add_argument('--vocab_save_dir', default=vocab_save_dir, type=str, help='data path')
    parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('-p', '--padding_size', default=200, type=int, help='Padding size of sentences.(default=128)')

    parser.add_argument('--model', default='cnn')

    parser.add_argument('-t', '--test_sample_percentage', default=0.1, type=float, help='The fraction of test data.(default=0.1)')
    parser.add_argument('-e', '--embed_size', default=512, type=int, help='Word embedding size.(default=512)')
    parser.add_argument('-f', '--filter_sizes', default='3,4,5', help='Convolution kernel sizes.(default=3,4,5)')
    parser.add_argument('-n', '--num_filters', default=128, type=int, help='Number of each convolution kernel.(default=128)')
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float, help='Dropout rate in softmax layer.(default=0.5)')
    parser.add_argument('-c', '--num_classes', default=95, type=int, help='Number of target classes.(default=18)')
    parser.add_argument('-l', '--regularizers_lambda', default=0.01, type=float, help='L2 regulation parameter.(default=0.01)')
    parser.add_argument('-b', '--batch_size', default=256, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate.(default=0.005)')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--fraction_validation', default=0.05, type=float, help='The fraction of validation.(default=0.05)')

    parser.add_argument('--results_dir', default=result_path, type=str,
                        help='The results dir including log, model, vocabulary and some images.(default=./results/)')

    parser.add_argument('--workers', default=32, type=int, help='use worker count')
    params = parser.parse_args()

    return params


