#IMPORTING THE NEEDED PACKAGES
import cv2
from tkinter import *
from tkinter import filedialog
import os
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk
from ttkthemes import themed_tk as theme
from PIL import ImageTk,Image
import imutils
import numpy as np
from imutils import contours
from imutils.object_detection import non_max_suppression
import argparse
import pytesseract
from skimage.segmentation import clear_border
import re
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

N = 9
d = []
a = []
grid = []
path = r"C:\Users\Karan Kapoor\OneDrive\Desktop\My Coding Projects\Sudoku Solver\digit_classifier.h5"
kernel = np.ones((1,1),np.uint8)
path2 = "C:\\Users\\Karan Kapoor\\OneDrive\\Desktop\\My Coding Projects\\Sudoku Solver\\SOLVEDSUDOKU.jpg"
path3 = "C:/Users/Karan Kapoor/OneDrive/Desktop/My Coding Projects/Sudoku Solver/SOLVEDSUDOKU.jpg"

model = load_model(path)
def disppuzzle(celllocsrow, grid, sudoku_image):
    for i in range(0,9):
        for j in range (0,9):
            cell = grid[i][j]
            cellloc = celllocsrow[i][j]
            cv2.putText(sudoku_image, str(cell), cellloc, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,5,250), 2)
    cv2.imshow("RESULT", sudoku_image)
def printing(arr):
    for i in range(N):
        for j in range(N):
            print(arr[i][j], end = " ")
        print()
def isSafe(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
    for x in range(9):
        if grid[x][col] == num:
            return False
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True

def solveSuduko(grid, row, col):
    if (row == N - 1 and col == N):
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return solveSuduko(grid, row, col + 1)
    for num in range(1, N + 1, 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num
            if solveSuduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False

def solvepuzzle(a):
    grid = []
    r1 = a[0:9]
    r2 = a[9:18]
    r3 = a[18:27]
    r4 = a[27:36]
    r5 = a[36:45]
    r6 = a[45:54]
    r7 = a[54:63]
    r8 = a[63:72]
    r9 = a[72:81]
    grid.append(r1)
    grid.append(r2)
    grid.append(r3)
    grid.append(r4)
    grid.append(r5)
    grid.append(r6)
    grid.append(r7)
    grid.append(r8)
    grid.append(r9)
    print(grid)
    return grid
def whatsthenumber(cnts, cell_thresh, mask, percentfilled):
    if percentfilled<0.03:
        number = 0
    else:
        digit = cv2.bitwise_and(cell_thresh, cell_thresh, mask = mask)
        roi = cv2.resize(digit, (28,28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        number = model.predict(roi).argmax(axis=1)[0]
    return number
def extract_number(y, x, cell):
    q = str(x) + str(y)
    q = "C:\\Users\\Karan Kapoor\\OneDrive\\Desktop\\My Coding Projects\\Sudoku Solver\\Sudoku Cells\\" + q + ".jpg"
    cell_thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cell_thresh = clear_border(cell_thresh)
    cell_thresh = cv2.erode(cell_thresh, kernel, iterations=1)
    cv2.imwrite(q, cell_thresh)
    cnts = cv2.findContours(cell_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        number = 0
        a.append(number)
    else:
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(cell_thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        (h,w) = cell_thresh.shape
        percentfilled = cv2.countNonZero(mask)
        number = whatsthenumber(cnts, cell_thresh, mask, percentfilled)
        a.append(number)
def sudoku_cellfn(sudoku_image, greysudoku_image):
    celllocsrow = [[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
                   [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
                   [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
                   [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
                   [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
                   [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
                   [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
                   [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)],
                   [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]]
    sudoku_board = np.zeros((9,9), dtype = 'int')
    step_x = greysudoku_image.shape[1] // 9
    step_y = greysudoku_image.shape[0] // 9
    location_cell = []
    for y in range(0, 9):
        row = []
        for x in range(0, 9):
            startX = x * step_x
            startY = y * step_y
            endX = (x + 1) * step_x
            endY = (y + 1) * step_y
            avg_x = (startX + endX)/2
            avg_y = (startY + endY)/2
            celllocsrow[y][x] = (int(avg_x), int(avg_y))
            row.append((startX, startY, endX, endY))
            cell = greysudoku_image[startY:endY, startX:endX]
            extract_number(y, x, cell)
    grid = solvepuzzle(a)
    if solveSuduko(grid, 0, 0):
        printing(grid)
        disppuzzle(celllocsrow, grid, sudoku_image)
    else:
        print("Invalid Puzzle")
def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped 

def exitfn():
    label6 = Label(root, text = "There seems to be an issue with the selected image. Use a different image").grid(row = 3, column = 7)

def openfn():
     print("Hello")
     global file_name
     global img_file
     file_name = filedialog.askopenfilename()
     image_sudoku = Image.open(file_name)
     image_sudoku2 = image_sudoku.resize((350,350),Image.ANTIALIAS)
     img_file = ImageTk.PhotoImage(image_sudoku2)
     label5 = Label(root,image = img_file).grid(row=3, column=2)

def primary_sudokufn():
    global model
    global a
    model = load_model(path)
    grid = []
    a = []
    print("Hello")
    n = 1
    locationimage = r"C:\Users\Karan Kapoor\OneDrive\Desktop\My Coding Projects\Sudoku Solver\Sudoku Cells" 
    image_use = cv2.imread(file_name)
    image_use = cv2.resize(image_use, (800,800))
    grey_img = cv2.cvtColor(image_use, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(grey_img, (7,7), 3)
    img_thresh = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img_thresh = cv2.bitwise_not(img_thresh)
    img_contour = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = imutils.grab_contours(img_contour)
    img_contour = sorted(img_contour, key=cv2.contourArea, reverse=True)
    contours_sudoku = None
    for cnts in img_contour:
        perimeter = cv2.arcLength(cnts, True)
        approx_perimeter = cv2.approxPolyDP(cnts, 0.02*perimeter, True)
        if len(approx_perimeter) == 4:
            contours_sudoku=approx_perimeter
            break
        if contours_sudoku is None:
            print("There's an issue")
            exitfn()
    img_copy = image_use.copy()
    cv2.drawContours(img_copy, [contours_sudoku], -1, (0,255,120), 2)
    sudoku_image = four_point_transform(img_copy, contours_sudoku.reshape(4,2))
    greysudoku_image = four_point_transform(grey_img, contours_sudoku.reshape(4,2))
    sudoku_cellfn(sudoku_image, greysudoku_image)
            
            
root = theme.ThemedTk()
root.get_themes()
root.set_theme("vista")
root.title("SUDOKU SOLVER")
root.geometry("1200x800")
label1 = ttk.Label(root, text="SUDOKU SOLVER", font=("Segoe UI Semilight",30)).grid(row=0, column=4)
label2 = ttk.Label(root, text="  ", font=("Segoe UI",30)).grid(row=1, column=3)
label3 = ttk.Label(root, text="     ", font=("Segoe UI",20)).grid(row=1, column=1)
button1 = ttk.Button(root, text="OPEN IMAGE", width=60, command=openfn).grid(row=2, column=2)
label4 = ttk.Label(root, text="      ", font=("Segoe UI",20)).grid(row=2, column=5)
button2 = ttk.Button(root, text="SOLVE!", width=60, command=primary_sudokufn).grid(row=2, column=7)
root.mainloop()