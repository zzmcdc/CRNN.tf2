export CUDA_VISIBLE_DEVICES=0,1,2,3 
python3 train.py -ta /data/recog/reg_imgs/train.txt  -va /data/recog/reg_imgs/val.txt --charset label_tf.txt --img_width 1024  --epochs 120


