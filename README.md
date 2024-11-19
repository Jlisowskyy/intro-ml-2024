# Introduction to Machine Learning 2024 - Project

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
    - [Project Structure](#project-structure)
3. [Usage](#usage)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
   - [Running the Project](#running-the-project)
       - [Frontend](#frontend)
       - [Testing](#testing)
       - [Scripts](#scripts)
4. [Project Description](#project-description)
5. [Authors](#authors)
6. [License](#license)

## Introduction:

This repository contains the code for the Introduction to Machine Learning 2024 project. 
The project task is to develop CNN model for intercom device detecting authorised and unauthorised people.

## Getting Started:

### Project Structure:

```
├── examples
├── reports
└── src
    ├── audio
    ├── cnn
    ├── frontend
    ├── helper_scripts
    ├── pipelines
    └── test
```

- `examples`: Contains example code using jupyter notebooks.
- `reports`: Contains reports and documentation created for project milestones etc.
- `src`: Contains the source code for the project.
    - `audio`: Contains the code for the audio processing part of the project.
    - `cnn`: Contains code for the CNN model.
  - `frontend`: Contains the code for the frontend of the project.
    - `helper_scripts`: Collection of standalone scripts used in the project.
    - `pipelines`: Contains the code for the data processing pipelines.
  - `test`: Contains the code for testing the project.

## Usage:

### Prerequisites:

To set up the project, ensure you have the following dependencies:

- Python 3.12 or higher
- Python virtual environment (`virtualenv`) for dependency management (recommended)

### Installation:

#### 1. First, clone the repository:

```shell
git clone https://github.com/Jlisowskyy/intro-ml-2024
````

#### 2. Navigate to the project directory:

```shell
cd intro-ml-2024
```

#### 3. Initialize the virtual environment:

```shell
python -m venv .venv
```

#### 4. Activate the virtual environment:

On Unix or macOS:

```shell
source .venv/bin/activate
```

On Windows:

```shell
.venv\Scripts\activate
```

#### 5. Install the required dependencies:

```shell
pip install -r requirements.txt
```

### Running the project:

Running the project is easy as never! Simply run:

```shell
python main.py
```

To start our CLI (interactive mode and argument mode) to get detailed description on running specific functionalities.

## Project Description:


## Authors:

- Łukasz Kryczka
- Michał Kwiatkowski
- Jakub Lisowski
- Tomasz Mycielski
- Kuba Pietrzak

## License:

Licensed under the MIT License. See `LICENSE` for more information.
