# began

Add a short description here!

## Installation

This package is supplied with a `conda` environment that will install all of the required packages:

```bash
conda env create -f environment.yaml --prefix=/path/to/project/envs
```

Activate with:

```bash
conda activate /path/to/project/envs
```

And install the package:

```bash
python setup.py install
```

## Comet software

Here we add some notes about running code on the GPU nodes of the Comet cluster. This will have both generally applicable tools (such as Jupyter lab being run on a compute node), and Comet-specific notes.

### Singularity

`Singularity` is a container tool for HPC environments, similar to Docker. This was originally convenient to use on Comet, as the most up-to-date CUDA libraries, required to support `tensorflow-2.0`, were not available, and `Singularity` could be used to download pre-made images from Docker with the correct software and run them on Comet. Since the update of 12/03/2019, this is no longer necessary as CUDA 10.1 will be the default library, which supports `tensorflow-2.0`. 

### Jupyter-Lab

First establish the Python environment to use, and ensure `jupyter-lab` is installed. In this project we have the `began` `conda` environment. 

#### Jupyter on interactive jobs

These instructions work as of 12/03/2019. For a similar set of instructions on Bridges, see [these notes](https://gist.github.com/mcburton/d80e4395cd82737d3677c570aa31ee40).

1. On Comet, request an interactive GPU allocation. 

2. On the GPU node check the host name using:

```bash
hostname -f
```

This will be used later to setup `ssh` forwarding from your local machine to the compute node.

3. Next run the `jupyter-lab` server, specifying the following options:

```bash
jupyter-lab --no-browser --ip=0.0.0.0
```

This should start the server running, and you will have been allocated a port, e.g.:

```bash
Notebook server running at http://127.0.0.1:8
```

4. Setup `ssh` forwarding from a local port to the port on the GPU node.



#### Jupyter as an `sbatch` job

#### Jupyter on login nodes

This is usually best avoided for computational work. 

### Tensorboard

### SSH

It is generally recommended to `ssh` to the SSO hub by the XSEDE manual:

```bash
ssh -l bthorne login.xsede.org
```

and then to `gsissh` into Comet:

```bash
gsissh comet
```

However, it will generally be easier for us to `ssh` directly into comet, in order to include the port forwarding we use to connect to servers being run on Comet:

```bash
ssh -l bthorne comet.sdsc.edu -L (port forwarding options)
```



## Tensorboard on remote

In order to display tensorboard for a remote run of the code we need to `ssh` into the machine and forward a local port to the port on the server we will be using. To do this use the `-L` flag with `ssh`:

```bash
ssh -L localPort:127.0.0.1:remotePort user@server
```

In our case this is:

```bash
ssh -L 16006:127.0.0.1:6006 bthorne@comet.sdsc.edu
```

and in the initiated `ssh` session I run:

```bash
tensorboard --logdir=$TF_LOGDIR
```

which will send the tensorboard service to the default port of 6006. If we wanted to use a different port we could, but this needs to match the port specified in the `ssh` connection.

## Description

A longer description of your project goes here...

## Installation

In order to set up the necessary environment:

1. create an environment `began` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate began
   ```
3. install `began` with:
   ```
   python setup.py install # or `develop`
   ```

Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
   ```
   pre-commit install
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

5. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yaml` for the exact reproduction of your
   environment with:
   ```
   conda env export -n began -f environment.lock.yaml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yaml` using:
   ```
   conda env update -f environment.lock.yaml --prune
   ```
## Project Organization

```
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── began               <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## Note

This project has been set up using PyScaffold 3.2.3 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
