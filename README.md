# Steam Recommender
A recommender system demo for the Steam gaming platform. Developed as a project in the course TNM108, at Linköping University.

Developers on this project:
- Frans Johansson
- Jonas Bertilson
- Moa Gutenwik

Please check the [project blog page](https://frans-johansson.github.io/steam-recommender/) for more information about the ideas, methods and data explored in this project.

## Try it yourself

The core functionality implemented in the codebase is demonstrated in the `demo.ipynb` Jupyter notebook file. This section will quickly walk you through the steps to try running the code yourself.

### Setting up the virtual environment

To run the code, you will need to install some prerequisite libraries. We recommend setting up a Python virtual environement for this. You can read more about virtual environments [here](https://docs.python.org/3/library/venv.html), but the general steps are as follows:

1. Make sure you are in the root directory of the source code, then run `python -m venv [enter the name of your virtual environment here]`. This should create a directory with the name of your virtual environment in the shell's current directory.
2. Run `source [name of your virtual environment directory]/bin/activate` (Note: if you are using the Fish shell, substitute `activate` with `activate.fish`).
3. You're in your venv! To exit out of the venv, type `deactivate`.

With you venv activated, you should be able to install the required libraries with `pip install -r requirements.txt` (again, making sure you are in the root directory of the project).

### Running the Jupyter notebook

To run the notebook file `demo.ipynb` you can either just open it directly in an editor that natively supports Jupyter notebooks. One quite excellent such editor is [Visual Studio Code](https://code.visualstudio.com/), where you should only really need the Python extension before being able to run `.ipynb` files directly. Just make sure to select the correct Python interpreter (you will want the one in your venv directory).

Alternatively you can always set up a local server to host the notebook. To do this, simply run `jupyter notebook` in the root directory of the project and open the url given in the terminal to start playing around with the notebook in your web browser of choice.
