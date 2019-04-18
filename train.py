import argparse
import logging
import torch
from dataset import mydataset
from base_predictor import Predictor
from torch.utils.data import DataLoader

def main(args):
    logging.info('preprocess labels...')
    data = mydataset(args.image_dir, args.label_dir)
    _valid = mydataset(args.valid_image_dir, args.valid_label_dir)
    predictor = Predictor(valid = _valid)
    predictor.fit_dataset(data)

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Script to train.')
    parser.add_argument('image_dir', type=str,
                        help='Directory to image directory.')
    parser.add_argument('label_dir', type=str,
                        help='Directory to label directory.')
    parser.add_argument('valid_image_dir', type=str,
                        help='Directory to valid image directory.')
    parser.add_argument('valid_label_dir', type=str,
                        help='Directory to valid label directory.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
