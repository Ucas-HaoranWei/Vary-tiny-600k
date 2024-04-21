<p align="center">
<img src="asset/vary-600k.jpg" style="width: 200px" align=center>
</p>
<p align="center">
<a href="">Vary-tiny-600k</a>       
</p>

## Background
- The [Huggingface](https://github.com/huggingface/transformers) version of Vary-tiny  suffers potential issues, leading to the loss being hard to converge under multiple epochs.
- Many friends are very interested in the train data of Vary. 


## Release
-  [2024/4/21] 🔥🔥🔥 We present a Vary-tiny LAVIS codebase and the Vary-600k dataset !!!


## Contents
- [Install](#install)
- [Train](#train)
- [Demo](#demo)
- [Vary-600k](#vary-600k)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data, code, and checkpoint are intended and licensed for research use only.


## Install
1. Clone this repository and navigate to the Vary-tiny-600k folder
```bash
git clone https://github.com/Ucas-HaoranWei/Vary-tiny-600k.git
cd LAVIS-main
```
2. Install Package
```Shell
pip install -e .
```
3. Prepare Pretrain Weights and Data
   - download the OPT-125M [here](https://huggingface.co/facebook/opt-125m/tree/main) and the SAM-b weights [here](https://github.com/facebookresearch/segment-anything)
   - download the Vary-600k [here](https://pan.baidu.com/s/18Rh53JxvbYYl9BPHoFvWcQ ) with code "vary"
   - prepare the dirs as follows:
   - 
   ![image](https://github.com/Ucas-HaoranWei/Vary-tiny-600k/assets/50487563/21d529ea-be53-41d3-9ca0-72eb29958bef)

## Train
```Shell
python -m torch.distributed.run --nproc_per_node=8 --master_port=29501 train.py --cfg-path lavis/projects/varytiny/train/pretrain.yaml
```
or multi machines
```Shell
python -m torch.distributed.run --master_addr xxx --master_port xxx --node_rank xxx --nnodes xxx --nproc_per_node xxx  train.py --cfg-path lavis/projects/varytiny/train/pretrain.yaml
```

If your training goes smoothly, your loss (end of each epoch) will be similar to the following (2×8 H800)：

   ![image](https://github.com/Ucas-HaoranWei/Vary-tiny-600k/assets/50487563/9c02a5a5-e93d-4a94-bd7d-c4b76d30d6f6)



## Demo
1. change the ``pretrained'' and ``finetuned'' path with your checkpoints in ``LAVIS-main/lavis/configs/models/varytiny/varytiny_inference.yaml'', such as:
2. 
   ![image](https://github.com/Ucas-HaoranWei/Vary-tiny-600k/assets/50487563/8c008c8f-862f-4e0d-afc5-6117d5e7a527)
```Shell
python tests/models/test_varytiny.py  --image-file  xxx.jpg
```
## Vary-600k
- Vary-600k is a PDF image-text pair dataset with about 30W English and 30W Chinese pages.
- The dataset is extracted using Fitz. A BERT model is used to merge sentences within paragraphs. Paragraphs are separated by "\<lb>". The reason why we do not use "\n" is because "\n" is the "EOS" of opt-125m.
- You can use Vary-600k for your pretrain, warm-up, and so on.
- Note that Vary-600k is only a sub-data of the pretrain data used in the original [Vary](https://github.com/Ucas-HaoranWei/Vary).
- Download Vary-600k [here](https://pan.baidu.com/s/18Rh53JxvbYYl9BPHoFvWcQ). Code: "Vary"

## Acknowledgement
- [LAVIS](https://github.com/salesforce/LAVIS): the codebase we built upon!


## Citation
If you find our work useful in your research, please consider citing Vary:
```bibtex
@article{wei2023vary,
  title={Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models},
  author={Wei, Haoran and Kong, Lingyu and Chen, Jinyue and Zhao, Liang and Ge, Zheng and Yang, Jinrong and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2312.06109},
  year={2023}
}

@article{wei2024small,
  title={Small Language Model Meets with Reinforced Vision Vocabulary},
  author={Wei, Haoran and Kong, Lingyu and Chen, Jinyue and Zhao, Liang and Ge, Zheng and Yu, En and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2401.12503},
  year={2024}
}
```


