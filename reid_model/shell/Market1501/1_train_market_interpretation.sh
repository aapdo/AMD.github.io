cd 
cd amd/reid_model

gpus='1'

# ps -ef | grep "python3 ./projects/InterpretationReID" | awk '{print $2}' | xargs kill
CUDA_VISIBEVICES=$gpus python3 ./projects/InterpretationReID/train_net.py  --config-file ./projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip.yml  --num-gpus 1  MODEL.DEVICE "cuda:0"