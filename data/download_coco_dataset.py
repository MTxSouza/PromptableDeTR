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
import sys

sys.path.append(os.path.abspath(path=os.path.dirname(p=__file__.split(sep=os.sep)[-2])))

import argparse
import asyncio
from dataclasses import dataclass
from enum import Enum

import httpx
from tqdm import tqdm

from utils.logger import get_logger

__author__ = "Matheus Oliveira de Souza (msouza.os@hotmail.com)"


# Global variables.
logger = get_logger(name="dataset_downloader", level="debug")


# Enums.
@dataclass
class DatasetMetadata:
	"""
	Metadata for each set.
	"""
	name: str
	filename: str
	url: str


class DatasetURL(Enum):
	"""
	Enum storing all URLs for each set.
	"""
	train = DatasetMetadata(name="train", filename="train_images.zip", url="http://images.cocodataset.org/zips/train2017.zip")
	valid = DatasetMetadata(name="valid", filename="valid_images.zip", url="http://images.cocodataset.org/zips/val2017.zip")
	test = DatasetMetadata(name="test", filename="test_images.zip", url="http://images.cocodataset.org/zips/test2017.zip")


# Functions.
def cli_args():
	"""
	Get all CLI arguments to be used in script.
	"""
	# Define parser.
	parser = argparse.ArgumentParser(prog="dataset_downloader", usage=__doc__)

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


async def download_file(url, output, client, pbar):
	"""
	Download a file from WEB and track it download progress 
	on CLI.

	Args:
		url (str): URL of file to be downloaded.
		output (str): Output path to store the file.
		client (httpx.AsyncClient): Async client object used to download files.
		pbar (tqdm.tqdm): Progress bar object to be updated and displayed on terminal.
	"""
	# Download file.
	async with client.stream(method="GET", url=url) as response:
		
		# Get total size of file.
		total_size = int(response.headers.get(key="content-length", default=0))
		pbar.total = total_size

		# Save file locally.
		with open(file=output, mode="wb") as file_buffer:
			async for chunk in response.aiter_bytes(chunk_size=1024):
				file_buffer.write(chunk)

				# Update progress bar.
				pbar.update(n=len(chunk))


async def async_progress_bar(name, url, output):
	"""
	Define a progress bar to be displayed on terminal during a file 
	downloading.

	Args:
		name (str): Name of the file to be downloaded.
		url (str): Dataset file URL to download.
		output (str): Output directory path to store the file.
	"""
	logger.debug(msg="Preparing progress bar for %s file..." % name)
	await asyncio.sleep(1.5)

	# Define output path.
	file_path = os.path.join(output, name)

	# Define progress bar.
	with tqdm(total=0, unit="B", unit_scale=True, desc=name, leave=True) as pbar:
		async with httpx.AsyncClient() as client:
			await download_file(url=url, output=file_path, client=client, pbar=pbar)


async def download_datasets(urls, output):
	"""
	Prepare the gather object that stores the download function 
	that downloads the dataset files.

	Args:
		urls (List[DatasetMetadata]): Tuple with all dataset file URLs to download.
		output (str): Output directory path to store the files.

	Returns:
		asyncio.gather: Gather object to be ran.
	"""
	# Define limit of concurrent downloads.
	semaphore = asyncio.Semaphore(value=3)

	async def semaphore_download(name, url, output):
		try:
			async with semaphore:
				await async_progress_bar(name=name, url=url, output=output)
		except Exception as error:
			logger.critical(msg="Unexpected error has been occurred during %s download: %s" % (name, str(error)))

	# Create download tasks.
	tasks = [semaphore_download(name=data.filename, url=data.url, output=output) for data in urls]

	return await asyncio.gather(*tasks)


def main():
	"""
	Main function to run the script.
	"""
	# Get CLI arguments.
	args = cli_args()

	# Get URLs.
	logger.debug(msg="Setting up all file URLs to be downloaded.")
	url_list = [DatasetURL["train"].value] # Default dataset to download.
	if not args.no_valid:
		url_list.append(DatasetURL["valid"].value)
	if not args.no_test:
		url_list.append(DatasetURL["test"].value)

	# Run tasks.
	asyncio.run(main=download_datasets(urls=url_list, output=args.output))


if __name__=="__main__":
	main()
