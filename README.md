# RIS-DMMI
This repository provides the PyTorch implementation of DMMI in the following papers:<br />
__Beyond One-to-One: Rethinking the Referring Image Segmentation (ICCV2023)__ <br />

# News
* 2023.10.03-The final version of our dataset has been released. Please remember to download the latest version.
* 2023.10.03-We release our code.

# Dataset
We collect a new comprehensive dataset Ref-ZOM (**Z**ero/**O**ne/**M**any), which contains image-text pairs in one-to-zero, one-to-one and one-to-many conditions. Similar to RefCOCO, RefCOCO+ and G-Ref, all the images in Ref-ZOM are selected from COCO dataset. Here, we provide the text, image and annotation information of Ref-ZOM, which should be utilized with COCO_trainval2014 together. <br />
Our dataset could be downloaded from:<br />
[[Baidu Cloud](https://pan.baidu.com/s/1CxPYGWEadHhcViTH2iI7jw?pwd=g7uu)] [[Google Drive](https://drive.google.com/drive/folders/1FaH6U5pywSf0Ufnn4lYIVaykYxqU2vrA?usp=sharing)] <br />
Remember to download original COCO dataset from:<br />
[[COCO Dowanload](https://cocodataset.org/#download)]<br />

# Code

**Prepare**<br />
* Download the COCO_train2014 and COCO_val2014, and merge the two dataset as a new folder “trainval2014”. Then, in the Line-52 in `/refer/refer.py`, give the path of this folder to `self.Image_DIR`<br />
* Download and rename the "Ref-ZOM(final).p" as "refs(final).p". Then put refs(final).p and instances.json into `/refer/data/ref-zom/*`.  <br />
* Prepare the Bert similar to [LAVT](https://github.com/yz93/LAVT-RIS)
* Prepare the Refcoco, Refcoco+ and Refcocog similar to [LAVT](https://github.com/yz93/LAVT-RIS)

**Train**<br />
* Remember to change `--output_dir` and `--pretrained_backbone` as your path.<br />
* Utilize `--model` to select the backbone. 'dmmi-swin' for Swin-Base and 'dmmi_res' for resnet-50.<br />
* Utilize `--dataset`, `--splitBy` and `--split` to select the dataset as follwos:<br />
```
# Refcoco
--dataset refcoco, --splitBy unc, --split val
# Refcoco+
--dataset refcoco+, --splitBy unc, --split val
# Refcocog(umd)
--dataset refcocog, --splitBy umd, --split val
# Refcocog(google)
--dataset refcocog, --splitBy google, --split val
# Ref-zom
--dataset ref-zom, --splitBy final, --split test
```
* Begin training!!<br />
```
sh train.sh
```

**Test**
* Remember to change `--test_parameter` as your path. Meanwhile, set the `--model`, `--dataset`, `--splitBy` and `--split` properly. <br />
* Begin test!!<br />
```
sh test.sh
```

# Parameter
**Refcocog(umd)**<br />
| Backbone  | oIoU | mIoU | Google Drive |Baidu Cloud |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ResNet-101  | 59.02  | 62.59  | [Link](https://drive.google.com/file/d/1ziDIeioglD08QQyL-_yGFFlao3PtcJJS/view?usp=drive_link)  | [Link](https://pan.baidu.com/s/1uKJ-Wu5TtJhphXNOXo3mIA?pwd=6cgb)  |
| Swin-Base  | 63.46  | 66.48  |  [Link](https://drive.google.com/file/d/1uuGWSYLGYa_qMxTlnZxH6p9FMxQLOQfZ/view?usp=drive_link)  |  [Link](https://pan.baidu.com/s/1eAT0NgkID4qXpoXMf2bjEg?pwd=bq7w)  |

**Ref-ZOM**<br />
| Backbone  | oIoU | mIoU | Google Drive |Baidu Cloud |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Swin-Base  | 68.77  | 68.25  |  [Link](https://drive.google.com/file/d/1Ut_E-Fru0bCmjtaC2YhgOLZ7eJorOOpi/view?usp=drive_link)  |  [Link](https://pan.baidu.com/s/1T-u55rpbc4_CNEXmsA-OJg?pwd=hc6e)  |

# Acknowledgements

We strongly appreciate the wonderful work of [LAVT](https://github.com/yz93/LAVT-RIS). Our code is partially founded on this code-base. If you think our work is helpful, we suggest you refer to [LAVT](https://github.com/yz93/LAVT-RIS) and cite it as well.<br />

# Citation
If you find our work is helpful and want to cite our work, please use the following citation info.<br />
```
@InProceedings{Hu_2023_ICCV,
    author    = {Hu, Yutao and Wang, Qixiong and Shao, Wenqi and Xie, Enze and Li, Zhenguo and Han, Jungong and Luo, Ping},
    title     = {Beyond One-to-One: Rethinking the Referring Image Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {4067-4077}
}

