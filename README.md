# BiDet
Source from https://github.com/ZiweiWangTHU/BiDet
## License

BiDet is released under the MIT License. See the LICENSE file for more details.

## Contact

If you have any questions about the code, please contact Ziyi Wu dazitu616@gmail.com


https://github.com/ZiweiWangTHU/BiDet/issues/27

```shell
python ssd/train_bidet_ssd.py --dataset=VOC --data_root=../arp_data/VOC2007/VOCdevkit/ --num_workers 16 --batch_size 32  --cuda 1
```

```shell
python ssd/eval_voc.py --weight_path=logs/VOC/model_25000_loc_1.2368_conf_1.9463_reg_0.0_prior_0.0_loss_3.1832_lr_0.001.pth --voc_root=../arp_data/VOC2007/VOCdevkit
```


```shell
python ssd/eval_coco.py --weight_path=logs/COCO/ --coco_root=../arp_data/
```

```shell
python ssd/train_bidet_ssd.py --dataset=COCO --data_root=../arp_data/COCO/ --num_workers 16 --batch_size 32  --cuda 1 --resume 1 --weight_path logs/COCO/model_10000_loc_2.2888_conf_2.9886_reg_0.0_prior_0.0_loss_5.2774_lr_0.001.pth   --start_iter 10000
```