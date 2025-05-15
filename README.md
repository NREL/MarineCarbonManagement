# MarineCarbonManagement
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


[![Link](https://img.shields.io/badge/Publication-Electrochemical_Direct_Ocean_Capture-brightgreen)](https://www.nrel.gov/docs/fy24osti/90673.pdf)

![Static Badge](https://img.shields.io/badge/SWR-24--122-purple)

The marine carbon management software is an open-source Python based software that contains generic models for marine carbon capture. More models are under development and will be added soon. 

## Software requirements

- Python version 3.9, 3.10, 3.11 64-bit
- Other versions may still work, but have not been extensively tested at this time

## Installing from Source

1. Using Git, navigate to a local target directory and clone repository:

    ```bash
    git clone https://github.com/NREL/MarineCarbonManagement.git
    ```

2. Navigate to `MarineCarbonManagement`

    ```bash
    cd MarineCarbonManagement
    ```

3. Create a new virtual environment and change to it. Using Conda and naming it 'mcm':

    ```bash
    conda create --name mcm python=3.11 -y
    conda activate mcm
    ```

4. Install MarineCarbonManagement and its dependencies:

    - If you want to just use MarineCarbonManagement:

       ```bash
       pip install .  
       ```

    - If you also want development dependencies for running tests:  

       ```bash
       pip install -e ".[develop]"
       ```

    - If you also want development dependencies for running tests:  

       ```bash
       pip install -e ".[examples]"
       ```

    - In one step, all dependencies can be installed as:

      ```bash
      pip install -e ".[all]"
      ```


7. Verify setup by running tests:

    ```bash
    pytest
    ```
