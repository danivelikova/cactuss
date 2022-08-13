# CACTUSS: Common Anatomical CT-US Space for US examinations
![cactuss_pipeline](https://user-images.githubusercontent.com/14922785/184475714-4c2a75f6-ef65-484a-9940-b44719e9a2f5.png)

CACTUSS projects generates a common anatomical space between CT and Ultrasound images. We provide our implementation for the second step of the CACTUSS pipeline, which is the Intermediate Representation (IR) translation step for any input B-mode image.

## Getting started

Follow the steps below to get the code running on your local machine.

## Installation

```
git clone https://github.com/danivelikova/cactuss.git
git submodule init
git submodule update
pip install -r requirements.txt
```

## Usage

After installing the requirements run the example script to generate the Intermediate Representations (IR).
```
python3 ir_test.py
```

This example will execute the following steps:
  
  1. Creates a folder checkpoints/ and downloads the model with pretrained weights in it
  2. Creates a folder datasets/ and downloads the test images
  3. Runs the inference script
  4. Creates a folder results/ where the resulting image will be stored
 
 
## Exemplary results
<img src="https://user-images.githubusercontent.com/105121035/167472824-bab7db1c-cbcf-4fa1-a38d-09b68adf6a38.png" width="256" height="800" />

