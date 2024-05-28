#cd pth_to_fast-reid-interpretation
cd 
cd amd/reid_model
#set gpus
gpus='0, 1, 2'
#train
CUDA_VISIBLE_DEVICES=$gpus python3 ./tools/train_net.py  --config-file ./configs/Market1501/bagtricks_circle_R50.yml  MODEL.BACKBONE.PRETRAIN_PATH  './pretrain_models/market_circle_r50_ip.pth'  MODEL.DEVICE "cuda:0"   MODEL.BACKBONE.WITH_NL  False   TEST.METRIC   "euclidean"   TEST.EVAL_PERIOD 10  SOLVER.CHECKPOINT_PERIOD 10
#CUDA_VISIBLE_DEVICES=$gpus python3 ./tools/train_net.py  --config-file ./projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip.yml  MODEL.BACKBONE.PRETRAIN_PATH  './pretrain_models/market_circle_r50_ip.pth'  MODEL.DEVICE "cuda:0"   MODEL.BACKBONE.WITH_NL  False   TEST.METRIC   "euclidean"   TEST.EVAL_PERIOD 10  SOLVER.CHECKPOINT_PERIOD 10