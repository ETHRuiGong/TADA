### This is the official implementation of our ECCV 2022 paper  ["TACS: Taxonomy Adaptive Cross-Domain Semantic Segmentation"](https://arxiv.org/pdf/2109.04813.pdf)

### Prerequisite
*  CUDA/CUDNN 
*  Python3
*  Packages found in requirements.txt

### Dataset Preparation
GTAV, SYNTHIA, Cityscapes, Synscapes

### Open Taxonomy Setting

Before Relabeling: python3 trainTACS_open_adduncertaincontrastive.py --config ./configs/configUDA_euler.json --name TACS_open_uncertaincontrast_beforerelabel --numsamples 30

After Relabeling: python3 trainTACS_open_adduncertaincontrastive_relabel.py --config ./configs/configUDA_euler_resume.json --name TACS_open_uncertaincontrast_afterrelabel --numsamples 30 --resume *\<Path to CheckPoint Before Relabeling\>*

### Coarse-to-Fine Taxonomy Setting
Before Relabeling: python3 trainTACS_coarsetofine_adduncertaincontrastive.py --config ./configs/configUDA_euler.json --name TACS_coarsetofine_uncertaincontrast_beforerelabel --numsamples 30

After Relabeling: python3 trainTACS_coarsetofine_adduncertaincontrastive_relabel.py --config ./configs/configUDA_euler_resume.json --name TACS_coarsetofine_uncertaincontrast_afterrelabel --numsamples 30 --resume *\<Path to CheckPoint Before Relabeling\>*

### Implicitly Overlapping Taxonomy Setting
Before Relabeling: python3 trainTACS_implicitoverlapping_adduncertaincontrastive.py --config ./configs/configUDA_euler.json --name TACS_implicitoverlapping_uncertaincontrast_beforerelabel --numsamples 15

After Relabeling: python3 trainTACS_implicitoverlapping_adduncertaincontrastive_relabel.py --config ./configs/configUDA_euler_resume.json --name TACS_implicitoverlapping_uncertaincontrast_afterrelabel --numsamples 15 --resume *\<Path to CheckPoint Before Relabeling\>*

### Model Testing ###

Open, Coarse-to-Fine: python3 evaluateTACS.py --model-path *\<Path to Checkpoint\>*

Implicitly-Overlapping: python3 evaluateTACS_16classes.py --model-path *\<Path to Checkpoint\>*

### Acknowledgements

The implementation is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [DACS](https://github.com/vikolss/DACS)
* [ContrastiveSeg](https://github.com/tfzhou/ContrastiveSeg)

### Citation

If this helps you, please cite our TACS work:

```
@inproceedings{gong2022tacs,
  title={TACS: Taxonomy Adaptive Cross-Domain Semantic Segmentation},
  author={Rui Gong, Martin Danelljan, Dengxin Dai, Danda Pani Paudel, Ajad Chhatkuli, Fisher Yu, Luc Van Gool},
  booktitle={ECCV},
  year={2022}
}
```

