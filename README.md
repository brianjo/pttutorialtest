# PyTorch Tutorials Experiments

PyTorch tutorials are presented as sphinx style documentation at:

## [https://pytorch.org/tutorials](https://pytorch.org/tutorials)

## This project

This project is designed to test pytorch tutorials one at a time.

First, run `pip install -r requirements.txt` to get the theme and prerequisits.  

Next, drop a properly formatted .py file in beginner_source and then run:

```
make html 
```
Your files will be built into `_builds/html`. You can test with Python by running `python -m http.server` from the `_builds\html` folder. The file your testing will be located at server:8000/beginner/filename.html.


