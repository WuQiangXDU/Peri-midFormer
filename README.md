# Peri-midFormer: Periodic Pyramid Transformer for Time Series Analysis (NeurIPS 2024 Spotlight)

Qiang Wu, Gechang Yao, Zhixi Feng and Shuyuan Yang, "Peri-midFormer: Periodic Pyramid Transformer for Time Series Analysis", NeurIPS, 2024.

## Abstract

<p align="center">
<img src="https://github.com/WuQiangXDU/Peri-midFormer/blob/main/figures/Architecture.jpg" width="600" class="center">
</p>

Time series analysis finds wide applications in fields such as weather forecasting, anomaly detection, and behavior recognition. Previous methods attempted to model temporal variations directly using 1D time series. However, this has been quite challenging due to the discrete nature of data points in time series and the complexity of periodic variation. In terms of periodicity, taking weather and traffic data as an example, there are multi-periodic variations such as yearly, monthly, weekly, and daily, etc. In order to break through the limitations of the previous methods, we decouple the implied complex periodic variations into inclusion and overlap relationships among different level periodic components based on the observation of the multi-periodicity therein and its inclusion relationships. This explicitly represents the naturally occurring pyramid-like properties in time series, where the top level is the original time series and lower levels consist of periodic components with gradually shorter periods, which we call the periodic pyramid. To further extract complex temporal variations, we introduce self-attention mechanism into the periodic pyramid, capturing complex periodic relationships by computing attention between periodic components based on their inclusion, overlap, and adjacency relationships. Our proposed Peri-midFormer demonstrates outstanding performance in five mainstream time series analysis tasks, including short- and long-term forecasting, imputation, classification, and anomaly detection.


## Platform

- NVIDIA 4090 24GB GPU, PyTorch

## Usage

1. Install Python 3.8. For convenience, execute the following command.

````
pip install -r requirements.txt
````

2. Prepare Data. You can obtain the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), then place the downloaded data in the folder ````./dataset````.

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder ````./scripts/.```` You can reproduce the experiment results as the following examples:
````
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/PerimidFormer_ETTh2.sh
# short-term forecast
bash ./scripts/short_term_forecast/PerimidFormer_M4.sh
# imputation
bash ./scripts/imputation/ECL_script/PerimidFormer.sh
# anomaly detection
bash ./scripts/anomaly_detection/SMD/PerimidFormer.sh
# classification
bash ./scripts/classification/PerimidFormer_EthanolConcentration.sh
````

## Acknowledgement
We greatly appreciate the following GitHub repositories for their valuable code:

- https://github.com/thuml/Time-Series-Library
