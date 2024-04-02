#cd pth_to_fast-reid-interpretation
#cd /export/home/cxd/fast-reid-interpretation-1008

#set gpus
gpus='1'

#train
CUDA_VISIBLE_DEVICES=$gpus 


python /home/workspace/AMD/projects/InterpretationReID/train_net.py --config-file /home/workspace/projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip.yml --eval-only MODEL.WEIGHTS /home/workspace/export/home/cxd/fast-reid-interpretation-1008/logs/market1501/bagtricks_circle_R50/model_final.pth MODEL.DEVICE "cuda:0"

#python ./projects/InterpretationReID/train_net.py  --config-file ./projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip.yml --eval-only MODEL.WEIGHTS /export/home/cxd/fast-reid-interpretation-1008/logs/market1501/bagtricks_circle_R50/model_final.pth MODEL.DEVICE "cuda:0"   MODEL.BACKBONE.WITH_NL  False   TEST.METRIC   "euclidean"   TEST.EVAL_PERIOD 10 

#python visualize_result.py --config-file /home/workspace/logs/market1501/bagtricks_circle_R50/config.yaml --parallel --vis-label --dataset-name 'Market1501' --output logs/market_1948 --opts MODEL.WEIGHTS /export/home/pretrain_models/market_circle_r50_ip.pth