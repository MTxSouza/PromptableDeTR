"""
This script will just install locally all necessary utilities to run the project.
"""
import nltk

if __name__=="__main__":
    nltk.download(info_or_id="averaged_perceptron_tagger_eng")
    nltk.download(info_or_id="punkt_tab")
