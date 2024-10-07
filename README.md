# Introduction to Machine Learning 2024 - Project

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
    - [Project Structure](#project-structure)
3. [Usage](#usage)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Project Description](#project-description)
5. [Authors](#authors)
6. [License](#license)

## Introduction:

This repository contains the code for the Introduction to Machine Learning 2024 project. 
The project task is to develop CNN model for intercom device detecting authorised and unauthorised people.

## Getting Started:

### Project Structure:

├── examples
├── reports
└── src
    └── frontend

- `examples`: Contains example code using jupyter notebooks.
- `reports`: Contains reports and documentation created for project milestones etc.
- `src`: Contains the source code for the project.
    - `frontend`: Contains the code for the frontend of the project.

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

### Frontend:

Simply run:

```shell
fastapi dev src/frontend/main.py
```

## Project Description:


## Authors:

- Jakub Lisowski
- Michał Kwiatkowski
- Łukasz Kryczka
- Kuba Pietrzak
- Tomasz Mycielski

## License:

Licensed under the MIT License. See `LICENSE` for more information.
