### Introduction

This is an implementation of the spiking cerebellar granular layer model described in:

> Taylor R Fore et al. "Acetylcholine modulates cerebellar granule cell spiking by regulating the balance of synaptic excitation and inhibition"
> Journal of Neuroscience 1 April 2020, DOI: https://doi.org/10.1523/JNEUROSCI.2148-19.2020

### Platform information

**Platform:** osx-64

**Python:** 3.6.8

**Brian:** 2.2.1

**Numpy:** 1.15.4

**Pandas:** 0.24.1

**Scipy:** 1.2.0

**Matplotlib:** 3.0.2

The model is implemented using the [Brian simulator](http://briansimulator.org/) for spiking neural networks, which is written in Python.

Data processing and visualization are done using Numpy, Pandas, Scipy, and Matplotlib.

### Package installation

To install the required packages, type in terminal:

```
pip install --user PACKAGE_NAME
```

To ensure reproducibility of results in paper, use the same package versions as above.

```
pip install --user brian2
pip install --user numpy==1.15.4
pip install --user scipy==1.2.0
pip install --user pandas==0.24.1
pip install --user matplotlib==3.0.2
```

Alternatively using Anaconda, you can use the specification file provided to create an identical conda environment to run the scripts by typing in terminal:

```
conda create --name myenv --file requirements.txt
```

This 'requirements.txt' file was created on the osx-64 platform, and may not work correctly on others. See [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details.

### Code repository

This folder contains two python scripts for running the simulation and generating the figures in the paper:

- **run_simulation.py:** Main script to run single instantiation of the model.
- **figures.py:** Data analysis and visualization

### Running the scripts

To run the provided scripts, navigate to the relevant folder by typing in terminal:

```
cd FOLDER_NAME
```

The simulation of all three conditions (control, GABAzine, muscarine) is built into the main script. To run, type in terminal:

```
python run_simulation.py
```

The code saves all relevant variables as numpy arrays to the current directory for further analysis. Robust progress reporting is output to the console, but expect the entire simulation to take approximately three hours (depending on computer specs).

To run analysis and generate the figures from the paper, type in terminal:

```
python figures.py
```

Figure windows will not appear, but pdf files of each figure will be saved in the working directory.
