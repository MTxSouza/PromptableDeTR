# How to train

To train the **PromptableDeTR** model, you'll need to download the pre-trained weights of the encoders and some other necessary files to use the main model. Check the `Base weights` page on `Releases`.

You can use the bash script `train.sh` to deploy the training, this script has already some hyper-parameters defined, but, you must need to specify the following:
|Parameter|Description|
|-|-|
|`TRAIN_DATASET_DIR`|Path to the directory where the training samples are located.|
|`IMAGE_DIR`|Path to the directory where the images are located.|
|`IMG_ENC_WEIGHT`|Path to the weights of the image encoder.|
|`TXT_ENC_WEIGHT`|Path to the weights of the text encoder.|
|`VOCAB_FILE`|Path to the vocab file of the text encoder.|

You can easily run the script using the command below:
```bash
$ bash TRAIN_DATASET_DIR=<...> IMAGE_DIR=<...> IMG_ENC_WEIGHT=<...> TXT_ENC_WEIGHT=<...> VOCAB_FILE=<...> train.sh
```
> The `.sh` script will first run the `install_utilities.py` script to install all dependencies for training, and after training is over, it'll delete all weight files stored at experiment directory, keeping only the best weights for you, whereas the `.py` will only run the main training script.

During training, it will create an experiment directory where all weights and training logs will be stored, the `.log` files will contains some predictions of the model, so you be able to monitoring the optimization of the model during time.