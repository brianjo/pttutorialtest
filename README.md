# PyTorch Tutorials Experiments

PyTorch tutorials are presented as sphinx style documentation at:

## [https://pytorch.org/tutorials](https://pytorch.org/tutorials)

This project is designed to test PyTorch tutorials one at a time. Our PyTorch tutorials site can take several hours to build, making it difficult for a tutorial writer to test their tutorials in a timely manner. This project strips away the downloads and tutorial content to make it easy to edit and test new tutorials locally. 

The PyTorch project uses Sphinx-Gallery's notebook styled [examples](https://sphinx-gallery.readthedocs.io/en/latest/tutorials/plot_notebook.html#sphx-glr-tutorials-plot-notebook-py) to create the tutorials that reside on pytorch.org. The syntax is very simple. In essence, you write a specially  formatted Python file and it shows up as documentation page. You can read instructions for creating a Sphinx-Gallery file [here](https://sphinx-gallery.readthedocs.io/en/latest/syntax.html).

If you prefer to write your tutorial in Jupyter, you can use this [script](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe) to convert the notebook to Python file. 

To use this project for testing, fork this project and clone it locally: 
```
git clone <your fork path>
cd pttutorialtest
pip install -r requirements.txt # gets the theme and prerequisits 
```

Next, drop a properly formatted .py file in beginner_source. The file must use follow the <name>_tutorial.py pattern. (The example in the repo is named example_tutorial.py.) Finally run the build:

```
make docs 
```
Your files will be built into `_builds/html`. You can test with Python by running `python -m http.server` from the `_builds\html` folder. The file your testing will be located at server:8000/beginner/tutorialname_tutorial.html.

## Using GitHub Actions

As an alternative to building locally you can do the following. 

* Fork this repo.
* Drop a new tutorial in beginner_source.

There's a GitHub action that will build the site and put the results  in your forks github-pages branch. You'll need to enable this to use it. After it's enabled, the site will be generated to a gh-pages branch. If you have gh-pages enabled, you can view the site at https://`<username>`.github.io/pttutorialtest/

You'll need to navigate to the page you generate directly if you want to see the
output. In this repo, I have an example called example_tutorial.py and the url
ends up being beginner/example_tutorial.html you can see the results below. 

https://brianjo.github.io/pttutorialtest/beginner/example_tutorial.html

Note that any new tutorial you're testing needs to end with _tutorial.py for it to
be built. 
