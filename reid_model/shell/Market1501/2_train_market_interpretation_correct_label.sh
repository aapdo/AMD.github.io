cd 
cd amd/reid_model

gpus='0'

# ps -ef | grep "python3 ./projects/InterpretationReID" | awk '{print $2}' | xargs kill
CUDA_VISIBEVICES=$gpus python3 ./projects/InterpretationReID/train_net.py  --config-file ./projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip_correct_label.yml  MODEL.DEVICE "cuda:0"
# CUDA_VISIBEVICES='0' --config-file ./projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip.yml  MODEL.DEVICE "cuda:0"