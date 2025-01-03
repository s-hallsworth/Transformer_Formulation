﻿# Optimising over Transformers

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#features">Features</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#workflow">Workflow</a></li>
        <li><a href="#license">License</a></li>
        <li><a href="#how-to-contribute">How to Contribute</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project implements a mathematical formulation of a trained Transformer Neural Network (TNN), enabling integration into optimisation pipelines. This allows for TNNs to be used in decision making processes whereby the optimal input conditions that lead to a desired predictive outcome can be determined. This is unlike traditional achine learning approaches which predict a future outcome from some given input data.

This code creates a mixed-integer non-linear program (MINLP) formulation of a trained TNN, in order to format the data-driven model in a way that is suitable for global optimisation. The formulation preserves the exactness of the trained model so that predictions achieved by the formulated TNN match the trained TNN's results. Various bounds and cuts can be applied to the improve the solving efficiency of the formulation. Nevertheless, it is only possible to  optimise over relatively small TNNs in a reasonable amount of time. More information about this Master's thesis project can  be found in the associated thesis document found at https://repository.tudelft.nl/.

The code for generating the formulation can be found in the "MINLP_tnn" folder and three examples are included in the "examples" folder. The examples include:
* A verification problem using an encoder-only vision TNN (Pytorch based) 
* A simple optimal trjectory problem using an encoder-only TNN (Keras based)
* A reactor optimisation problem using an encoder-decoder TNN (HuggingFace based)



### Features
- **Mathematical Formulation of TNN**: \
  Implements a trained Transformer Neural Network (TNN) in mathematical form for optimisation purposes.
- **Togglable Bounds and Cuts**: \
  Includes options to enable or disable additional bounds and cuts for enhanced performance and flexibility.
- **Cross-Framework Compatibility**:\
   Supports conversion of TNN models trained in,
  - **PyTorch**
  - **Keras**
  - **HuggingFace**
- **Dual Optimisation Environments**: 
  - Built in **Pyomo** for broad solver compatibility.
  - Convertible to **Gurobipy** to leverage Gurobi's latest global optimisation capabilities for MINLPs.
- **Integration with Existing NN Formulations**:
  - Interfaces with **OMLT** (Pyomo) for Feed Forward Neural Networks.
  - Interfaces with **GurobiML** (Gurobipy) for Feed Forward Neural Networks.
- **Solver Flexibility**: \
  Provides access to multiple solvers via Pyomo and the cutting-edge features of Gurobi for robust optimisation.


### Built With
- [Python](https://www.python.org/) - Core development language.
- [Pyomo](http://www.pyomo.org/) - Optimisation framework.
- [Gurobipy](https://www.gurobi.com/products/gurobi-optimiser/) - Optimisation framework.
- [PyTorch](https://pytorch.org/) - For training and exporting TNNs.
- [Keras](https://keras.io/) - For training and exporting TNNs.
- [HuggingFace](https://huggingface.co/) - Transformer architecture support.
- [OMLT](https://github.com/cog-imperial/OMLT) - Feed Forward Neural Network formulations for Pyomo.
- [GurobiML](https://www.gurobi.com/documentation/9.1/examples/gurobipy.html) - Feed Forward Neural Network formulations for Gurobi.

<!-- This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url] -->


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
To build the MINLP_tnn package, run the following command in the root folder
  ```sh
  pip install -e .
  ```

To set up to run the associated examples, you can follow to steps below:

1. Create virtual environment 'pyomo_env'
   ```sh
   python -m venv pyomo_env
   ```
2. Activate virtual environment \
   Linux:
   ```sh
   source pyomo_env\Scripts\activate
   ```
   Windows:
   ```sh
   pyomo_env\Scripts\activate
   ```
3. Install required python libraries \
   There are two requirements files, the main reason for this is to change the tensorflow version needed for the reactor example.
   * "requirements.txt" --> tensforflow v2.15.0 
   * "requirements_reactor.txt" --> tensforflow v2.17.0. Required for forked gurobiML code used to implement SiLU activation for reactor case study.
   ```sh
   python -m pip install -r .\requirements.txt
   ```
4. [OPTIONAL] For the reactor case study, custom versions of HuggingFace's timeseries transformer and GurobiML's feed-forward Neural Network are used. This custom versions can be accessed as follows.
   * Retrieve and build time series transformer in reactor folder:
    ```sh
   git clone https://github.com/s-hallsworth/transformers.git
   cd transformers 
   pip install -e .
   cd ..
   ```
   * Retrieve GurobiML with SiLU activation function in reactor folder:
   ```sh
   git clone https://github.com/s-hallsworth/gurobi-machinelearning.git gurobi_machinelearning
   ```
5. Install GUROBI license tools [here](https://www.gurobi.com/downloads/gurobi-software/)
6. Create a GUROBI account and request a license
7. Install GUROBI license
   ```sh
   grbgetkey <your_license_key>
   ```
8. Follow the [instructions](https://portal.ampl.com/user/ampl/request/amplce) to download AMPL 
9.  Download the pre-compiled version of SCIP following [instructions](https://scipopt.org/#download). SCIP requires the Visual C++ Redistributable Packages. Ensure SCIP is installed on the global path
  


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
### Workflow
Overall, the code is set up to work as shown in the image below. The associated python files for each step are mentioned on the side. The repository includes a file called “example_notebook.ipynb” which identifies each of the functions needed to set up a new problem.

<p align="center">
  <img src=".\Examples\images\setup_overview.png" alt="Workflow Overview" width="450">
</p>

The example case studies can be run by going to the associated folder and running the "run_exp" python file. For example:
```sh
python run_exp_veri.py
```
This will run the verification experiment with various set ups and save the log files which can be visualised using gurobi_logtools.

### License
This project is licensed under the MIT License.


### How to Contribute

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add my feature"`.
4. Push to your branch: `git push origin feature/my-feature`.
5. Open a Pull Request.

<p align="right">(<a href="#readme-top">back to top</a>)</p

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

<!-- Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet) -->
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template/blob/master/README.md)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
