# Peri-midFormer: Periodic Pyramid Transformer for Time Series Analysis

# TLCE
# Temporal Local Correntropy Representation for Fault Diagnosis of Machines
This repository is the implementation of our paper: 'Temporal Local Correntropy Representation for Fault Diagnosis of Machines', which has been published on IEEE TII. It is available at <https://ieeexplore.ieee.org/document/10061348>.

Abstract— In view of the good correlation measurement ability of correntropy, we propose a temporal local correntropy representation (TLCE) method based on the local correntropy matrix for fault diagnosis of machines. In TLCE, a sample is divided into several segments, and then the correlation between these segments is expressed by correntropy. Finally, the correntropy matrix composed of the correntropy is regarded as the feature of each sample. The proposed TLCE model is validated by experiments of three bearing datasets and one gear dataset. And results demonstrate that compared with other methods, TLCE has obvious advantages, such as effectiveness and robustness.


## Platform

- PyTorch and NVIDIA 4090 24GB GPU

## Datasets

You can download processed datasets at <>

## Usage

Install Python 3.8. For convenience, execute the following command.

````
pip install -r requirements.txt
````


## Citation

If you find this work helpful, please cite our paper:


 @ARTICLE{10061348,
  author={Feng, Zhixi and Wu, Qiang and Yang, Shuyuan},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Temporal Local Correntropy Representation for Fault Diagnosis of Machines}, 
  year={2023},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/TII.2023.3253180}}

     
 Thanks for your attention!
