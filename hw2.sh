# TODO: create shell script for running your YoloV1-vgg16bn model
wget -O baseline.pkl https://www.dropbox.com/s/pi06ogf6ynr1kqr/map_13066_yolo_epoch58.pkl?dl=0 
python3.7 predict.py ${1} ${2}
