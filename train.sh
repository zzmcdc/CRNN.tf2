export CUDA_VISIBLE_DEVICES=7
python3 train.py -ta /data/recog/reg_imgs/train.txt  -va /data/recog/reg_imgs/val.txt --charset label_tf.txt --img_width 256  --epochs 300 -lr 0.1


