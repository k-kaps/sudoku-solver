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
import sys
from skimage.segmentation import clear_border
import re
from PIL import Image
import imutils
N = 9

def replace_chars(text):
    """
    Replaces all characters instead of numbers from 'text'.
    
    :param text: Text string to be filtered
    :return: Resulting number
    """
    list_of_numbers = re.findall(r'\d+', text)
    result_number = ''.join(list_of_numbers)
    return result_number

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
 
def openfn():
     print("Hello")
     global file_name
     global img_file
     file_name = filedialog.askopenfilename()
     image_sudoku = Image.open(file_name)
     image_sudoku2 = image_sudoku.resize((350,350),Image.ANTIALIAS)
     img_file = ImageTk.PhotoImage(image_sudoku2)
     label5 = Label(root,image = img_file).grid(row=3, column=2)

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
  
def processfn():
    a = []
    b = []
    grid = []
    print("Hello")
    n = 1
    locationimage = r"C:\Users\Karan Kapoor\OneDrive\Desktop\My Coding Projects\Sudoku Solver\Sudoku Cells" 
    image_use = cv2.imread(file_name)
    image_use = imutils.resize(image_use, width = 600)
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
    img_copy = image_use.copy()
    cv2.drawContours(img_copy, [contours_sudoku], -1, (0,255,120), 2)
    cv2.imshow("Contours",img_copy)
    sudoku = four_point_transform(image_use, contours_sudoku.reshape(4,2))
    sudoku_grey = four_point_transform(grey_img, contours_sudoku.reshape(4,2))
    sudoku_blur = cv2.GaussianBlur(sudoku_grey, (7,7), 3)
    sudoku_thresh = cv2.adaptiveThreshold(sudoku_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
    sudoku_thresh = cv2.bitwise_not(sudoku_thresh)
    sudoku_contours = cv2.findContours(sudoku_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sudoku_contours = sudoku_contours[0] if len(sudoku_contours) == 2 else sudoku_contours[1]
    for c in sudoku_contours:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(sudoku_thresh, [c], -1, (0,0,0), -1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    sudoku_thresh = cv2.morphologyEx(sudoku_thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    sudoku_thresh = cv2.morphologyEx(sudoku_thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)
    sudoku_invert = 255 - sudoku_thresh
    sudoku_contours = cv2.findContours(sudoku_invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sudoku_contours = sudoku_contours[0] if len(sudoku_contours) == 2 else sudoku_contours[1]
    (sudoku_contours, _) = contours.sort_contours(sudoku_contours, method="top-to-bottom")
    sudoku_rows = []
    row = []
    for (i, c) in enumerate(sudoku_contours, 1):
        area = cv2.contourArea(c)
        if area < 50000:
            row.append(c)
            if i % 9 == 0:  
                (sudoku_contours, _) = contours.sort_contours(row, method="left-to-right")
                sudoku_rows.append(sudoku_contours)
                row = []
    for row in sudoku_rows:
        for c in row:
            mask = np.zeros(sudoku.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)
            result = cv2.bitwise_and(sudoku, mask)
            result[mask==0] = 255
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result_thresh = cv2.threshold(result_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            result_thresh = clear_border(result_thresh)
            conts = cv2.findContours(result_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conts = imutils.grab_contours(conts)
            if len(conts) != 0:
                print("Not less than 0")
                c2 = max(conts, key=cv2.contourArea)
                mask = np.zeros(result_thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c2], -1, 255, -1)
                (h, w) = result_thresh.shape
                result_thresh = cv2.bitwise_and(result_thresh, result_thresh, mask = mask)
            name = "sudokuimg" + str(n) + ".jpg"
            cv2.imwrite(os.path.join(locationimage, name), result_thresh)
            pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Karan Kapoor\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
            digits = pytesseract.image_to_string(result_thresh, config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789")
            print("{}\n".format(digits))
            digits = "".join([c if ord(c) < 128 else "" for c in digits]).strip()
            b.append(digits)
            n = n + 1
            cv2.waitKey(10)
    while i<=81:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Karan Kapoor\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        imname = "sudokuimg" + str(i) + ".jpg"
        dirname = os.path.join(locationimage, imname)
        numbers = pytesseract.image_to_string(cv2.imread(dirname), config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789")
        print("{}\n".format(numbers))
        digits = "".join([c if ord(c) < 128 else "" for c in numbers]).strip()
        a.append(numbers)
        i = i+1
    print(a)
    print(b)
    cv2.imshow("Result",sudoku_invert)
    
root = theme.ThemedTk()
root.get_themes()
root.set_theme("vista")
root.title("SUDOKU SOLVER")
root.geometry("1200x800")
label1 = ttk.Label(root, text="SUDOKU SOLVER", font=("Segoe UI Semilight",30)).grid(row=0, column=4)
label2 = ttk.Label(root, text="  ", font=("Segoe UI",30)).grid(row=1, column=3)
label3 = ttk.Label(root, text="     ", font=("Segoe UI",20)).grid(row=1, column=1)
button1 = ttk.Button(root, text="OPEN IMAGE", width=60, command=openfn).grid(row=2, column=2)
button2 = ttk.Button(root, text="SOLVE!", width=60, command=processfn).grid(row=2, column=7)
label4 = ttk.Label(root, text="      ", font=("Segoe UI",20)).grid(row=2, column=5)
root.mainloop()