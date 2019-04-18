import os
import argparse
import logging
import torch
from dataset2 import test_dataset
from base_predictor2 import Predictor
from torch.utils.data import DataLoader

def main(args):
    logging.info('loading test data...')
    predictor = Predictor()
    data = test_dataset(args.image_dir, args.store_label_dir)
    output_decode = predictor.predict_dataset(data)
    logging.info('write test labels...')
    filenames = data.filenames
    classes = ["plane", "ship", "storage-tank", "baseball-diamond",
    "tennis-court", "basketball-court", "ground-track-field", "harbor",
    "bridge", "small-vehicle", "large-vehicle", "helicopter",
    "roundabout", "soccer-ball-field", "swimming-pool", "container-crane"]

    for boxes, filename in zip(output_decode, filenames):
        img_name = os.path.basename(filename)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        txt = open(args.store_label_dir + '/' + label_name, 'w+')
        for box in boxes:
                box[:4] *= (512/448)
                txt.write("%f %f %f %f %f %f %f %f %s %f\n" % (
                        float(box[0].floor()),
                        float(box[1].floor()),
                        float(box[2].floor()),
                        float(box[1].floor()),
                        float(box[2].floor()),
                        float(box[3].floor()),
                        float(box[0].floor()),
                        float(box[3].floor()),
                        classes[int(box[5].int())],
                        float(box[4])
                        ))
        txt.close()
def _parse_args():
    parser = argparse.ArgumentParser(
        description='Script to train.')
    parser.add_argument('image_dir', type=str,
                        help='Path to image directory.')
    parser.add_argument('store_label_dir', type=str,
                        help='Path to stroe label directory.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
