# TODO: create shell script for running your YoloV1-vgg16bn model
wget -O best.pkl https://www.dropbox.com/s/zqwxxc3brdvvavc/map_13200_best_epoch72.pkl?dl=0 
python3.7 predict2.py ${1} ${2} 
