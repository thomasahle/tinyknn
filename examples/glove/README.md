# GloVe Dataset Preparation

This repository contains a script, prepare-dataset.sh, which helps you download the GloVe Twitter dataset and convert it into a suitable format for further processing.

## Usage

To use prepare-dataset.sh, follow these steps:

1. Ensure that the prepare-dataset.sh script is executable. In your terminal, navigate to the directory containing the script and run the following command:
```bash
chmod +x prepare-dataset.sh
```
2. Execute the script:
```bash
./prepare-dataset.sh
```
The script will download the GloVe Twitter dataset, create a dataset directory, and extract the dataset files into it. It will then convert the glove.twitter.27B.100d.txt file to a binary format and save it as glove.twitter.27B.100d.dat. Additionally, the script will generate a NumPy file named glove.twitter.27B.100d.npy.

## Customization
If you want to process other GloVe dataset files, you can modify the ./convert.py command in the prepare-dataset.sh script to include the desired dataset file names. For example:

```bash
./convert.py 200 glove.twitter.27B.200d.txt
```
