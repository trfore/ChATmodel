# ChAT Model

[![Linux](https://github.com/trfore/chatmodel/actions/workflows/build-linux64.yml/badge.svg)](https://github.com/trfore/chatmodel/actions/workflows/build-linux64.yml)
[![MacOS & Win x64](https://github.com/trfore/chatmodel/actions/workflows/build-osx64-win64.yml/badge.svg)](https://github.com/trfore/chatmodel/actions/workflows/build-osx64-win64.yml)

## Introduction

This is an implementation of the spiking cerebellar granular layer model described in:

> Taylor R Fore et al. "Acetylcholine modulates cerebellar granule cell spiking by regulating the balance of synaptic excitation and inhibition"
> Journal of Neuroscience 1 April 2020, DOI: https://doi.org/10.1523/JNEUROSCI.2148-19.2020

### Code repository

The model is implemented using the [Brian2 simulator](http://briansimulator.org/) for spiking neural networks, which is written in Python. The two scripts for running the simulation and generating the figures in the paper are:

- **run_simulation.py** - main script to run single instantiation of the model.
- **figures.py** - data analysis and visualization.

## How to Use

### Requirements

```
brian2==2.2.1
matplotlib
numpy>=1.19
pandas
python>=3.6.8,<3.8.0
scipy
scikit-learn
```

You can install these dependencies using either `conda-lock.yml` (recommended), `environment.yml` or `requirements.txt` file.

1. Install [miniconda] (recommended), [conda], or [mamba]
   - We highly recommend changing conda's default solver to `libmamba`, see: https://www.anaconda.com/blog/conda-is-fast-now
2. Clone the repo and `cd` into the directory.
3. Install the dependencies into virtual environment, i.e. `brian2`

```sh
# clone the repo
$ git clone https://github.com/trfore/chatmodel.git
$ cd chatmodel

# Recommended, create 'brian2' environment
$ conda install conda-lock
$ conda-lock install --name brian2 lock_files/conda-lock.yml

# create env using environment.yml
$ conda env create -f environment.yml
# create env using requirements.txt
$ conda create --name brian2 --file requirements.txt --channel conda-forge
```

We recommend using the `conda-lock.yml` file, as it will install validated package versions tested against the following platforms.

### Tested Platforms & Simulation Run Times

- Linux (Debian 10+, Ubuntu 20.04/22.04)
- macOS (10.13, 11, 12)
- Windows (10)

#### Simulation Run Times and Resource Utilization

- The full simulation takes ~ 42 minutes (2020 AMD 5600x, 32 GB) to 5 hours (2013 Intel I5-4258U, 8 GB) to run. This time is dependent on your system hardware.
- Unconstrained, the model uses a maximum of `27 GB` of RAM and `2` physical cores, but will run on systems with less RAM as the simulation will write to `cache` and `swap` at the cost of a longer run time.
- The output files, `*.npy`, are collectively `328 MB` in size. An archive is available on the release page (link: [v1.1.0](https://github.com/trfore/chatmodel/releases/tag/v1.1.0)).

### Running the scripts

Clone the repo and `cd` into the directory.

```sh
$ git clone https://github.com/trfore/chatmodel.git
$ cd chatmodel
```

All three drug conditions (control, GABAzine, muscarine) are built into the main script. To run the full simulation:

```sh
$ python run_simulation.py
```

The code saves all relevant variables as numpy arrays to the current directory for further analysis. Robust progress reporting is output to the console, but expect the entire simulation to take approximately 1 to 3 hours (depending on computer specs, see 'Tested Platforms & Simulation Run Times' for details).

To run analysis and generate the figures from the paper, type in terminal:

```sh
$ python figures.py
```

Figure windows will not appear, but pdf files of each figure will be saved in the working directory.

## Environment Variables

You can change experiment parameters by setting environment variables, for example:

```sh
# Run the simulation for 50 trials
$ export CHAT_NUM_TRIAL=50
$ python run_simulation.py
```

| Environment Variable   | Default | Description                                               | Required |
| ---------------------- | ------- | --------------------------------------------------------- | -------- |
| CHAT_NUM_MF            | 315     | int, number of mossy fiber inputs                         | No       |
| CHAT_NUM_GRC           | 4096    | int, number of granule cells                              | No       |
| CHAT_NUM_GOC           | 27      | int, number of golgi cells                                | No       |
| CHAT_NUM_TRIAL         | 100     | int, number of trials for the simulation                  | No       |
| CHAT_RANDOM_WEIGHTS    | True    | boolean, use random weights for synaptic connections      | No       |
| CHAT_SEED_TRIAL        | 451     | int, seed value for trial runs                            | No       |
| CHAT_SEED_WEIGHT_MEANS | 35      | int, seed value for generating the random synaptic weight | No       |

## Additional Notes

The paper was created using the following packages:

```sh
brian2==2.2.1
numpy==1.15.4
scipy==1.2.0
pandas==0.24.1
matplotlib==3.0.2
scikit-learn=0.19.2
```

The original `requirements.txt` file is available here: [e69794d](https://github.com/trfore/chatmodel/commit/e69794d5f8bfb676a317b3f1624e47d9baaaad4e)

---

# Authors

- Nathan Taylor (https://github.com/taylorbn)
- Taylor Fore (https://github.com/trfore)

## Maintainers

- Taylor Fore

---

# References

- https://briansimulator.org/
- https://github.com/brian-team/brian2
- https://github.com/trfore/chatmodel

## Anaconda & Mamba

- https://www.anaconda.com/download/
- https://docs.conda.io/en/latest/miniconda.html
- https://mamba.readthedocs.io/en/latest/installation.html
- https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

[conda]: https://www.anaconda.com/download/
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[mamba]: https://mamba.readthedocs.io/en/latest/installation.html
