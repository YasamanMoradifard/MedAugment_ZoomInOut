# MedAugment_ZoomInOut
Further Development of MedAugment method in Medical Data Augmentation. Project has been done in summer semester 2024.


# Zoom-In/Out Augmentation: Refining MedAugment for Medical Image Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository hosts the code and documentation for my advanced deep learning and medical image augmentation project (SS 2024). This work builds upon the foundational **MedAugment** framework by introducing novel **Random and Label-Aware Zoom-In/Zoom-Out** techniques to simulate shifting clinical focus and address data scarcity in Medical Image Analysis (MIA).

## Acknowledgments and Baseline Work
This project is built upon the work proposed in the paper:
**"MedAugment: Universal Automatic Data Augmentation Plug-in for Medical Image Analysis"** by *Zhaoshan Liu, Qiujie Lv, Yifan Li, Ziduo Yang, and Lei Shen*.
* **Original Paper:** [arXiv:2306.17466](https://arxiv.org/abs/2306.17466)
* **Acknowledgment:** I implemented the base MedAugment framework from their paper as the foundation for my experiments.

---

## My Contribution: Zoom-In & Zoom-Out Augmentation
While MedAugment provides an excellent, training-free approach to augmenting medical data without compromising crucial features, I hypothesized that incorporating controlled "shifting focus" mechanisms could further enhance model robustness. 

I introduced two novel augmentation methods into the pipeline:

1. **Random Resizing & Cropping (Zoom-In/Out):**
   * **Resizing:** Adjusting image scale within a specified range, altering the aspect ratio.
   * **Cropping:** Extracting a random rectangular portion, introducing high variability and potentially useful negative cases.
2. **Label-Aware Resizing & Cropping:**
   * **Label-Aware Resizing:** Resizing the image while mathematically ensuring that regions of interest (labeled areas/masks) remain safely within the frame.
   * **Label-Aware Cropping:** Cropping the image to simulate a physician "zooming in" on an anomaly, guaranteeing that critical pathological areas are never excluded.

### Results Highlight
These augmentations were tested across multiple architectures for both Classification (VGGNet, ResNeXt, ConvNeXt) and Segmentation (UNet++, FPN, DeepLabV3). 
* **Overall Impact:** Nearly 40% of the experiments showed improvement over the standard MedAugment baseline.
* **Standout Performance:** The Label-Aware Zoom-In strategy showed notable improvements in segmentation tasks. For instance, **DeepLabV3 improved by 3.64%** and **FPN improved by 1.0%** compared to using MedAugment alone.

---

## Repository Structure

```
├── classification.py          # Main training loop and arguments for classification tasks
├── segmentation.py            # Main training loop and arguments for segmentation tasks
├── utils/                     
│   ├── equipment.py           # Device configuration (CPU/GPU allocation)
│   ├── evaluation.py          # Metrics calculation (SoftIoU, confusion matrix, etc.)
│   └── __init__.py            # Initialization for utils module
├── Notebooks/
│   ├── My_experiement.ipynb   # Exploratory notebook implementing the custom Zoom-In/Out logic
│   ├── Running_code.ipynb     # Automated training and evaluation runs
│   └── Segmentation_Code.ipynb# TensorFlow/Keras specific implementations for label-aware resizing
├── requirements.txt           # Python dependencies required to run the codebase
└── README.md                  # Project documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/MedAugment-ZoomInOut.git](https://github.com/yourusername/MedAugment-ZoomInOut.git)
   cd MedAugment-ZoomInOut
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Core dependencies include: PyTorch, torchvision, segmentation-models-pytorch, albumentations, scikit-learn, and OpenCV.*

---

## Running the Code

### 1. Classification
To run a classification experiment, use the `classification.py` script. You can specify the model, dataset, and hyperparameter indices.

```bash
python classification.py --model VGGNet --dataset busi --index 5 --epoch 40 --lr 0.002
```
*Supported Models:* `VGGNet`, `ResNeXt`, `ConvNeXt`
*Supported Datasets:* `busi`, `lung`, `btmri`, `cataract`

### 2. Segmentation
To run a segmentation experiment, use the `segmentation.py` script. The script uses a SoftIoU loss function optimized for medical masks.

```bash
python segmentation.py --model DeepLab --dataset covid --index 5 --epoch 40 --bs 128
```
*Supported Models:* `UNetPP`, `FPN`, `DeepLab`
*Supported Datasets:* `LUNG`, `kvasir`, `cvc`, `covid`

---

## Future Work
While the label-aware cropping proved highly effective for preserving target areas and providing global context, future iterations of this work aim to:
* Apply these label-aware mechanisms to larger, more diverse datasets (e.g., MRI or CT volumes).
* Dynamically combine the strengths of the MedAugment spatial space with my label-aware Zoom-In approach during the same epoch.
* Evaluate computational constraints over extended epochs to fully realize the generalization capacity.

## Author
**Yasaman Moradi Fard** *MSc in Medical Engineering, Friedrich-Alexander-Universität Erlangen-Nürnberg* Project Representation Learning SS 2024
```
