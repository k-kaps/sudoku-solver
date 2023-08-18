import cv2
import pytesseract
a=[]
image = cv2.imread("C:/Users/Karan Kapoor/OneDrive/Desktop/My Coding Projects/Sudoku Solver/Sudoku Puzzles/Sudoku20.jpg")
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Karan Kapoor\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
sudoku = pytesseract.image_to_string(image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
print("{}\n".format(sudoku))
sudoku = "".join([c if ord(c) < 128 else "" for c in sudoku]).strip()
print(sudoku)
a.append(sudoku)
print(a)