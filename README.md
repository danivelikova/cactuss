# CACTUSS: Common Anatomical CT-US Space for US examinations
![cactuss](https://user-images.githubusercontent.com/105121035/167461418-bb33b22e-bc61-4e77-89cf-7b5948cae74c.png)
CACTUSS projects generates a common anatomical space between CT and Ultrasound images. We provide our implementation for the second step of the CACTUSS pipeline, which is the Intermediate Representation (IR) generation step for any input Bmode images.

## Getting started

Follow the steps below to get the code running on your local machine.

## Installation

```
git clone https://github.com/cactuss-ct-us/cactuss.git
git submodule init
git submodule update
pip install -r requirements.txt
```

## Usage

After installing the requirements run the example script to generate the Intermediate Representations (IR)
```
python3 ir_test.py
```

This example will execute the following steps:
  
  1. Create a folder checkpoints and download the model with pretrained weights in it
  2. Create a folder datasets and download the test images
  3. Run the inference script
  4. Create a folder results where the resulting image will be stored
 
 
## Exemplary results
<img src="https://user-images.githubusercontent.com/105121035/167472824-bab7db1c-cbcf-4fa1-a38d-09b68adf6a38.png" width="256" height="800" />

