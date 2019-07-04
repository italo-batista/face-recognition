## Before you start

Make sure you have python, OpenFace and dlib installed. You can either [install them manually](https://cmusatyalab.github.io/openface/setup/) or use a preconfigured docker image that has everying already installed:

#### Installings openface manually

```
git clone https://github.com/cmusatyalab/openface.git 
cd openface
python setup.py install 
```
Download dlibs models with: `./models/get-models.sh`


## Step 1

Make a folder called `/training-images/` somewhere on your computer.

## Step 2

Make a subfolder for each person you want to recognize. For example:

- `/training-images/will-ferrell/`
- `/training-images/chad-smith/`
- `/training-images/jimmy-fallon/`

## Step 3

Copy all your images of each person into the correct sub-folders

## Step 4

Run the openface scripts from inside the openface root directory:

First, do pose detection and alignment:

`./util/align-dlib.py ./../ia-ia-oh/data/train/ align outerEyesAndNose ./../ia-ia-oh/data/aligned-images/ --size 96`

Second, generate the representations from the aligned images:

`./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/`

When you are done, the `./generated-embeddings/` folder will contain a csv file with the embeddings for each image.
