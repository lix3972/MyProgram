python -m visdom.server
http://localhost:8097

python train.py --dataroot ./datasets/crowd_2 --name crowd_2_cyclegan --model cycle_gan --resize_or_crop none

#参考  python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan 

#参考  python train.py --dataroot ./datasets/apple2orange --name ap2or_cyclegan --model cycle_gan 


