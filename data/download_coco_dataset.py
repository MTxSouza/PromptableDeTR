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


def cli_args():
	"""
	Get all CLI arguments to be used in script.
	"""
	# Define parser.
	parser = argparse.ArgumentParser(prog="dataset_downloader")

	# Arguments.
	parser.add_argument("--output", "-o", type=str, required=True, help="Path to save the downloaded dataset.")
	parser.add_argument("--no-valid", action="store_true", help="Disable the download for validation set.")
	parser.add_argument("--no-test", action="store_true", help="Disable the download for test set.")

	args = parser.parse_args()

	# Validate arguments.
	# --output
	if not os.path.isdir(s=args.output):
		raise "Invalid value for --output. It must be a real directory to store the dataset."

	return args


async def download_dataset_file():
	# TODO: create function that downloads a file given 
	# an URL.
	pass


def main():
	"""
	Main function to run the script.
	"""
	# Get CLI arguments.
	args = cli_args()


if __name__=="__main__":
	main()
