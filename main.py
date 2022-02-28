import argparse
from data_loader.data_loaders import *
from train_test import *
from parse_config import ConfigParser
import pickle


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def main(config):
    device, deviceids = prepare_device(config['n_gpu'])
    print(device)
    print(deviceids)
    data = load_data(config, device)
    train(data, device, config)
    test(data[2], config)
    #infer(data[3], config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Interest')

    parser.add_argument('-c', '--config', default="./config.yaml", type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(parser)
    main(config)