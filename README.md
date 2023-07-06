
# MNIST Evens/Odds Classifier



## Installation

Python >= 3.11.0, CUDA >= 11.3

For installation, please first download the MNIST dataset with the provided script.

```bash
  sh download_data.sh
```

Then, download all package requirements with 

```bash
  pip install requirements.txt
```




## Demo

To train and test the model, please use the `--train` and `--test` arguments with `run.py`: 

```bash
  python run.py --dataset ./data/ --train --test
```

If testing with a pretrained checkpoint, be sure to specify a checkpoint path with the argument `--checkpoint`. Other config options can be seen in `run.py`. 

To run inference on a single image, please run 

```bash
  python demo.py --image_path IMAGE_PATH --checkpoint CHECKPOINT_PATH --data_dir ./data/
```


## Acknowledgements

 - https://jimut123.github.io/ for providing a simple but well performing CNN architecture for MNIST
