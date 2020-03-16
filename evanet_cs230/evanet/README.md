# Evolving Space-Time Neural Architectures for Videos

This repository (forked from [google_research](https://github.com/google-research/google-research)) contains the code and pretrained models for EvaNet:

    "Evolving Space-Time Neural Architectures for Videos"
    AJ Piergiovanni, Anelia Angelova, Alexander Toshev, and Michael S. Ryoo
    ICCV 2019

[arXiv](https://arxiv.org/abs/1811.10636).

        @inproceedings{evanet,
              title={Evolving Space-Time Neural Architectures for Videos},
              booktitle={International Conference on Computer Vision (ICCV)},
	      author={AJ Piergiovanni and Anelia Angelova and Alexander Toshev and Michael S. Ryoo},
	      year={2019}
	}


This code supports inference with an ensemble of models pretrained on Kinetics-400.
An example video is included in the data directory. The video is from HMDB [1]
corresponding to a cricket activity. Running the full evaluation on the Kinetics-400 
validation set available in November 2018 (roughly 19200 videos) gives 77.2% accuracy.

## iTGM Layer
The iTGM layer, a 3D version of the [TGM layer](https://github.com/piergiaj/tgm-icml19) from our ICML 2019 paper [Temporal Gaussian Mixture Layer for Videos](https://arxiv.org/abs/1803.06316) is in the tgm_layer.py file. This layer inflates a 2D spatial kernel based on a mixture of 1D (temporal) Gaussians. This allows the modeling of spatio-temporal filters with significantly fewer parameters.

## Installation and Running

To install requirements:

```bash
pip install -r evanet/requirements.txt
```

Then download the [model weights](https://drive.google.com/file/d/13JRSFIlYinKABnhFKTwB95kCwPUX5t8u/view) and place them in data/checkpoints.

To evalute the pre-trained EvaNet ensemble on a sample video:
```bash
python -m evanet.run_evanet --checkpoints=rgb1.ckpt,rgb2.ckpt,flow1.ckpt,flow2.ckpt
```

# Results

## Kinetics-400
These results are on the video available November 2018, about 10% less than the original dataset.

| Method | Accuracy |
| ------------- | ------------- |
| I3D | 72.6 |
| (2+1)D I3D | 74.3 |
| iTGM I3D | 74.4 |
| ResNet-50 (2+1)D | 72.1 |
| ResNet-101 (2+1)D | 72.8 |
| 3D Ensemble | 74.6 |
| iTGM-Ensemble | 74.7 |
| Diverse Ensemble (3D, (2+1)D, iTGM) | 75.3 |
| Two-stream I3D | 72.6 |
| Two-stream S3D-G | 76.2 |
| ResNet-50 + Non-local | 73.5 |
| Arch. Ensemble (I3D, ResNet-50, ResNet-101) | 75.4 |
| Top 1 (Individual, ours) | 76.4 |
| Top 2 (Individual, ours) | 75.5 |
| Top 3 (Individual, ours) | 75.7 |

## HMDB (3 splits)

| Method | Accuacy |
| ------------- | ------------- |
| Two-stream | 59.4 |
| Two-stream+IDT | 69.2 |
| R(2+1)D | 78.7 |
| Two-stream I3D | 80.9 |
| PoTion | 80.9 |
| Dicrim. Pooling | 81.3 |
| DSP | 81.5 |
| Top model (Individual, ours) | 81.3 |
| 3D-Ensemble | 79.9 |
| iTGM-Ensemble | 80.1 |
| EvaNet (Ensemble, ours) | 82.3 |


## Charades

|  Method | mAP (%) |
| ------------- | ------------- |
| Two-Stream + LSTM  | 17.8  |
| Async-TF  | 22.4  |
| TRN | 25.2 |
| Non-local NN | 37.5 |
| 3D-Ensemble (baseline) | 35.2 |
| iTGM-Ensemble (baseline) | 35.7 |
| Top 1 (individual, ours) | 37.3 |
| Top 2 (individual, ours) | 36.8 |
| Top 3 (individual, ours) | 36.6 |
| EvaNet (ensemble, ours) | 38.1 |


## Moments-in-Time

| Method | Accuracy |
| ------------- | ------------- |
| I3D | 29.5 |
| ResNet-50 | 30.5 |
| ResNet-50 + NL | 30.7 |
| Arch. Ensemble (I3D, ResNet-50, ResNet-101) | 30.9 |
| Top 1 (Individual, ours) | 30.5 |
| EvaNet (Ensemble, ours) | 31.8 |


## References:

[1] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. HMDB: A Large Video Database for Human Motion Recognition. ICCV, 2011
