# PointNet

An implementation of [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf) in PyTorch.

## Dataset Used

[ModelNet10](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip), this ZIP file contains CAD models from the 10 categories used to train the deep network.

## File Descriptions

- `model.py` - Model Architecture for the PointNet model
- `loss.py` - Custom loss function specific for the PointNet training process
- `dataset.py` - Custom dataset for ModelNet10

## Dependencies
- PyTorch 1.4.0
- Python 3.7.6
