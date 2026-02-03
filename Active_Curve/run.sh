lr_init:
Gaussian 0.0001
Rest 0.0002

Trainer.py:
python3 Trainer.py --train_data Design_chair --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 123 --pth_name Design_chair/models/QuantUNetT_chair --round_sampling 1
python3 Trainer.py --val_data ../InShaPe_dataset/densebench30k_pre --vis_path QuantUNetT_chair --batch_size 2 --gpu 1 --seed 123 --pth_name Design_chair/models/QuantUNetT_chair --round_sampling 233 --evaluate

Scanner.py:
python3 Scanner.py --beamshape Tear --gpu 1 --batch_size 20 --pth_name QuantUNetT_tear --round_sampling 1000 --vis_path Design_tear

Batched fitting:
python3 Scanner.py --beamshape Tear --gpu 1

NED shell script:

start from scratch:
nohup bash Neural_Experimental_Design.sh ring 0.0002 1 1000 1 10 > NED_ring.txt 2>&1 &
resume from breakpoint:
nohup bash Neural_Experimental_Design.sh ring 0.0002 {the_next_expected_sampleID} 1000 1 10 > NED_ring.txt 2>&1 &

Train UNet-T from scratch:
python3 train_unet.py --data tmp_trainingset_ring --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 123 --pth_name NED_unet_ring.pth.tar
And evaluate:
python3 train_unet.py --data tmp_trainingset_ring --batch_size 2 --gpu 1 --seed 123 --pth_name NED_unet_ring.pth.tar --val_vis_path NED_unet_ring --eval