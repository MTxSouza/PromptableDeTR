"""
This script downloads the COCO dataset (Common Object in Context) from 2017 and store it
locally. This is the main dataset used for PromptVision project and all data used to train, 
evaluate and test the model came from it.

To run the script, you can simply run the following command:
```
$ python download_coco_dataset.py --output-dir <path-to-dir>
```
It will automatically download all three sets (train, validation and test) and store inside 
the output path specified by `--output-dir` tag.
"""
import os
import argparse
import subprocess


__author__ = "Matheus Oliveira de Souza (msouza.os@hotmail.com)"


if __name__=="__main__":
	pass
