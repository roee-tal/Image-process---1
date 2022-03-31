# ImageProcessing Ex1
**Roee tal**

**ID-315858506**

**This is the first task in the course - the main purpose of this exercise is to get  acquainted with Python’s basic syntax and some of its image processing facilities**

### Version and platform
- Python Version - Python 3.8
- Platform - Pycharm

### Submission Files
ex1_main.py - The driver code.
This file is the main script for running the code I wrote. That is testing that each function that was required , was implemented by me successfully.

ex1_utils.py- The primary implementation class.
This file contains the implementation iv'e made for every function in the task(except gamma).

gamma.py- gamma correction - Question 4.6  


### Functions
**imReadAndConvert**
Reads an image, and returns the image converted as requested

**imDisplay**
Reads an image as RGB or GRAY_SCALE and displays it

**transformRGB2YIQ**
Converts an RGB image to YIQ color space

**transformYIQ2RGB**
Converts an YIQ image to RGB color space

**cumSum_calc**
help function to calculate the cumsum (want to try it by myself)

**hsitogramEqualize**
Equalizes the histogram of an image

**quantizeImage**
Quantized an image in to **nQuant** colors

**Quant2Shape**
Help function to quantize an image with shape 2 in to **nQuant** colors - this function does the quantization

**find_first_z**
Help function to find the first borders - **according to the pixel's amount** - same amount in each part

**find_new_z**
 Help function to calculate the new borders using the formula from the lecture.
 
 **find_q**
 Help function to calculate the new q values for each border
 
 **gammaDisplay**
 GUI for gamma correction
 
 
