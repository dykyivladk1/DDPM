# DDPM Implementation

This repository contains the implementation of simple Denoising Diffusion Probabilistic Model. DDPM uses a principle of applying a noise to an image, and then learns how to reverse this action.

## Getting Started

Follow these steps to use this implementation:

### Prerequisites

Ensure you have Python installed on your system. This code is compatible with Python 3.9 and newer versions.

### Dataset

For training and testing the DDPM model, you'll need a dataset. I used CelebA dataset which you download using the following link:

[CelebA Link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

After downloading, place the dataset in an appropriate directory within the your project structure, such as "./data".

### Installation

1. **Clone the repository** to your local computer:

    ```
    git clone https://github.com/dykyivladk1/DDPM.git
    ```


2. **Install the required dependencies**. It's recommended to create and use a virtual environment:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Training model**

    To train a model for custom dataset, you can use the following command:
    
    ```
    python scripts/train.py --image_dir <image_dir> --device <device>
    ```

