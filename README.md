# README

# **"Real-Time Grasp Detection Using Convolutional Neural Networks" Reproduce**

(The model is uploaded but you can train better yourself if you have the time and the machine or if you are learning **PyTorch/ML**. Please bear in mind that you need to read and adapt to your needs some parts of the code. Feel free to open an issue if you need help. I will try to update README and comment on the code.)

This implementation is mainly based on the algorithm from Joseph Redmon and Anelia Angelova described in https://arxiv.org/abs/1412.3128.

The method uses an RGB image to find a single grasp. A deep convolutional neural network is applied to an image of an object and as a result, one gets the coordinates, dimensions, and orientation of one possible grasp.

The images used to train the network model are from **[Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp)**.

# **Installation**

This code was developed with Python 3.6 on Ubuntu 18.04. Python requirements can be installed by:

```bash
pip install -r requirements.txt
```

# **Datasets**

Currently, only the **[Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp)** is supported.

## **Cornell Grasping Dataset Preparation**

1. Download the and extract **[Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp)**.
2. Place the data to make the data folder like:

```bash
${robot-grasp-detection}
|-- data
`-- |-- cornell
    `-- |-- 01
        |   |-- pcd0100.txt
        |   |-- pcd0100cneg.txt
        |   |-- pcd0100cpos.txt
        |   |-- pcd0100d.tiff
        |   |-- pcd0100r.png
	|   ......
	|-- 02
	|-- 03
	|-- 04
	|-- 05
	|-- 06
	|-- 07
	|-- 08
	|-- 09
	|-- 10
        `-- backgrounds
            |-- pcdb0002r.png
            ......
```

# **Training**

Training is done by the `train.py` script. Some of the parameters that need to be adjusted during the training process have been sorted into script `opts.py`. You can directly change the value of each corresponding parameter in script `opts.py`. 

Some basic examples:

```bash
# Train network model on Cornell Dataset
python train.py --description training_example --dataset cornell --dataset-path <Path To Dataset> --use-rgb 1 --use-depth 0
```

Trained models are saved in `output/models` by default, with the validation score appended.

# **Evaluation/Visualisation**

Evaluation or visualisation of the trained networks is done using the `eval_resnet50.py` script. Some of the parameters that need to be adjusted during the evaluation process have been sorted into script `opts.py` (eval_model part). You can directly change the value of each corresponding parameter in script `opts.py` (eval_model part). 

Important flags are:

- `--iou-eval` to evaluate using the IoU between grasping rectangles metric.
- `--eval-vis` to plot the network output (predicted grasping rectangles).

For example:

```bash
python eval_resnet50.py --dataset cornell --dataset-path <Path To Dataset> --iou-eval --trained-network <Path to Trained Network> --eval-vis
```