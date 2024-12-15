## SdalsNet: Self-Distilled Attention Localization and Shift Network for Unsupervised Camouflaged Object Detection



### Train code:

```
python -m torch.distributed.launch --nproc_per_node=1 main.py --arch vit_small --patch_size 8 --data_path /data1/shoupeiyao/directory/dino/COD/ --output_dir /data1/shoupeiyao/workspace/sdalsnet/ --batch_size=16 --saveckp_freq 1 --epochs 100 --frozen 60
```

### Test  code:

```
python testall.py --epoch 0100
```
