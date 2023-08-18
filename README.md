# Sudoku Solver

This is the GitHub repository for the Sudoku Solver, a software that used tradiitional computer vision techniques to read a Sudoku from an image, and then uses the backtracking algorithm to solve the Sudoku.

## Images
<p align="center">
  <img src="/Sudoku Puzzles/Sudoku11.jpg" width="350" />
  <img src="/SOLVEDSUDOKU.jpg" width="350" />
</p>

<p align="center">
  Sudoku Solver: Raw Sudoku Image fed to the Solver(Left) and Solved Sudoku Output (Right)
</p>

## Dependencies and Installation
The project is based on Python (version 3.8.10) and requires the following libraries:
1. OpenCV (version 4.2.0)
2. Numpy (version 1.21.0)

For usage, the following steps must be followed:
1. Ensure that Python (version 3.8.10) is installed on the system. <br/>
   To install Python (version 3.8.10) the official steps can be followed here: [https://www.python.org/downloads/release/python-3810](https://www.python.org/downloads/release/python-3810/)

2. Install the dependencies by running the following commands:
```
pip install opencv-python==4.2.0
pip install numpy==1.21.0
```
3. Now, the repository can be cloned at a location of your choice by running the following commands
```
cd /path/to/the/desired/location
git clone https://github.com/k-kaps/sudoku-solver.git
```
4. Once python has been added to the PATH by following [this process](https://docs.python.org/3/using/windows.html#:~:text=On%20the%20first%20page%20of,pip%20for%20the%20package%20installer.) the project can be run using:
```
cd /path/to/the/install/location
python sudoku-solver/src/main.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/k-kaps/sudoku-solver/blob/main/LICENSE) file for details.
