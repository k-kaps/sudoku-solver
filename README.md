# Sudoku Solver

This is the GitHub repository for the Sudoku Solver, a software that used tradiitional computer vision techniques to read a Sudoku from an image, and then uses the backtracking algorithm to solve the Sudoku.

## Images
<p>
  <img src="/data/sudoku_solver_interface.jpg" width="350" />
</p>
<p align="center">
  Sudoku Solver: User Interface
</p>

<p align="center">
  <img src="/data/Sudoku Puzzles/Sudoku11.jpg" width="350" />
  <img src="/results/solved_sudoku.jpg" width="350" />
</p>

<p align="center">
  Sudoku Solver: Raw Sudoku Image fed to the Solver(Left) and Solved Sudoku Output (Right)
</p>

## Dependencies and Installation
The project is based on Python (version 3.8.10) and requires the libraries specified in the requirements.txt file. <br/>
<br/>
For usage, the following steps must be followed:
<br/>

1. Ensure that Python (version 3.8.10) is installed on the system. To install Python (version 3.8.10) the official steps can be followed here: [https://www.python.org/downloads/release/python-3810](https://www.python.org/downloads/release/python-3810/)

2. Now, the repository can be cloned at a location of your choice by running the following commands
```
cd /path/to/the/desired/location
git clone https://github.com/k-kaps/sudoku-solver.git
```
3. Once cloned, the dependencies can be installed using the following commands:
```
cd /path/to/the/install/location
pip install -r requirements.txt
```
4. If python has been added to the PATH variable by following [this process](https://docs.python.org/3/using/windows.html#:~:text=On%20the%20first%20page%20of,pip%20for%20the%20package%20installer.) the project can be run using:
```
cd /path/to/the/install/location
python sudoku-solver/src/main.py
```
If not added to the PATH, the alternative is to specify the location to the python.exe file instead of the 'python' variable.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/k-kaps/sudoku-solver/blob/main/LICENSE) file for details.
