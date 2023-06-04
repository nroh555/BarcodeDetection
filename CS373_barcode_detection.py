# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

def seperateArraysToRGB( px_array_r, px_array_g, px_array_b, image_width, image_height):
    new_array = [[[0 for c in range(3)] for x in range(image_width)] for y in range(image_height)]

    for y in range(image_height):
        for x in range(image_width):
            new_array[y][x][0] = px_array_r[y][x]
            new_array[y][x][1] = px_array_g[y][x]
            new_array[y][x][2] = px_array_b[y][x]
    return new_array

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
########################## Step 1 Functions #################################################
#
# Consists of:
#   Turning a RGB image to a Greyscale Image
#   Scaling an image to 0 and 255  
#   
####################################################################################
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # STUDENT CODE HERE
    for i in range (image_height):
        for j in range (image_width):
            rgb_value = pixel_array_r[i][j]*0.299 + pixel_array_g[i][j]*0.587 + pixel_array_b[i][j]*0.114
            greyscale_pixel_array[i][j] = round(rgb_value)
    
    return greyscale_pixel_array

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    [min_value, max_value] = computeMinAndMaxValues(pixel_array, image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if(max_value - min_value != 0):
                new_value = (pixel_array[i][j] - min_value) * (255/(max_value - min_value))
            else:
                new_value = (pixel_array[i][j] - min_value)
            pixel_array[i][j] = round(new_value)
    return pixel_array
    
    
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    max_value = 0
    min_value = 255
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] > max_value:
                max_value = pixel_array[i][j]
            if pixel_array[i][j] < min_value:
                min_value = pixel_array[i][j]
    return [min_value, max_value]

########################## Step 2 Functions #################################################
#
# Consists of:
#   Using the standard deviation method and the image gradient method
#   
####################################################################################

def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(image_height):
        for y in range(image_width):
            output[x][y] = 0.0
    for x in range(1,image_height-1):
        for y in range(1,image_width-1):
            output[x][y] = 0.0
            output[x][y] = (pixel_array[x-1][y-1]*-1.0+ pixel_array[x-1][y]*0.0+ pixel_array[x-1][y+1]*1.0+pixel_array[x][y-1]*-2.0+pixel_array[x][y]*0.0+pixel_array[x][y+1]*2.0+pixel_array[x+1][y-1]*-1.0+pixel_array[x+1][y]*0.0+pixel_array[x+1][y+1]*1.0)*0.125
            output[x][y] = float(abs(output[x][y]))

    return output

def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(image_height):
        for y in range(image_width):
            output[x][y] = 0.0
    for x in range(1,image_height-1):
        for y in range(1,image_width-1):
            output[x][y] = 0.0
            output[x][y] = (pixel_array[x-1][y-1]*1.0+ pixel_array[x-1][y]*2.0+ pixel_array[x-1][y+1]*1.0+pixel_array[x][y-1]*0.0+pixel_array[x][y]*0.0+pixel_array[x][y+1]*0.0+pixel_array[x+1][y-1]*-1.0+pixel_array[x+1][y]*-2.0+pixel_array[x+1][y+1]*-1.0)*0.125
            output[x][y] = float(abs(output[x][y]))
    return output

def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    padded = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(image_height):
        for y in range(image_width):
            padded[x][y] = 0.0
    for x in range(image_height-4):
        for y in range(image_width-4):
            mean = (pixel_array[x][y] + 
                    pixel_array[x+1][y]+ 
                    pixel_array[x+2][y]+
                    pixel_array[x+3][y]+
                    pixel_array[x+4][y]+
                    pixel_array[x][y+1]+
                    pixel_array[x+1][y+1]+
                    pixel_array[x+2][y+1]+
                    pixel_array[x+3][y+1]+
                    pixel_array[x+4][y+1]+
                    pixel_array[x][y+2]+
                    pixel_array[x+1][y+2]+
                    pixel_array[x+2][y+2]+
                    pixel_array[x+3][y+2]+
                    pixel_array[x+4][y+2]+
                    pixel_array[x][y+3]+
                    pixel_array[x+1][y+3]+
                    pixel_array[x+2][y+3]+
                    pixel_array[x+3][y+3]+
                    pixel_array[x+4][y+3]+
                    pixel_array[x][y+4]+
                    pixel_array[x+1][y+4]+
                    pixel_array[x+2][y+4]+
                    pixel_array[x+3][y+4]+
                    pixel_array[x+4][y+4])*(1/25)
            numerator = ((pixel_array[x][y] - mean) **2 + (pixel_array[x+1][y]- mean) **2 + (pixel_array[x+2][y]- mean) **2 + (pixel_array[x+3][y]- mean) **2 + (pixel_array[x+4][y]- mean) **2 +
                         (pixel_array[x][y+1]- mean) **2 + (pixel_array[x+1][y+1]- mean) **2 + (pixel_array[x+2][y+1]- mean) **2 + (pixel_array[x+3][y+1]- mean) **2 + (pixel_array[x+4][y+1]- mean) **2 +
                         (pixel_array[x][y+2]- mean) **2 + (pixel_array[x+1][y+2]- mean) **2 + (pixel_array[x+2][y+2]- mean) **2 + (pixel_array[x+3][y+2]- mean) **2 + (pixel_array[x+4][y+2]- mean) **2 +
                         (pixel_array[x][y+3]- mean) **2 + (pixel_array[x+1][y+3]- mean) **2 + (pixel_array[x+2][y+3]- mean) **2 + (pixel_array[x+3][y+3]- mean) **2 + (pixel_array[x+4][y+3]- mean) **2 +
                         (pixel_array[x][y+4]- mean) **2 + (pixel_array[x+1][y+4]- mean) **2 + (pixel_array[x+2][y+4]- mean) **2 + (pixel_array[x+3][y+4]- mean) **2 + (pixel_array[x+4][y+4]- mean) **2 )
            output = (numerator /25)**(1/2)
            output = float(abs(output))
            padded[x+1][y+1] = output
    return padded

########################## Step 3 Functions #################################################
#
# Consists of:
#   Using the gaussian filter method 
#   
####################################################################################

def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(image_height):
        for y in range(image_width):
            output[x][y] = 0.0
            
    padded = createInitializedGreyscalePixelArray(image_width+2, image_height+2)
    for x in range(image_height):
        for y in range(image_width):
            padded[x+1][y+1] = pixel_array[x][y]
    for x in range(1,image_height+1):
        padded[x][0] = padded[x][1]
        padded[x][-1] = padded[x][-2]
    for y in range(image_width + 2):
        padded[0][y] = padded[1][y]
        padded[-1][y] = padded[-2][y]
    for x in range(image_height):
        for y in range(image_width):
            output[x][y] = (padded[x][y]*1.0+ padded[x+1][y]*2.0+ padded[x+2][y]*1.0+padded[x][y+1]*2.0+padded[x+1][y+1]*4.0+padded[x+2][y+1]*2.0+padded[x][y+2]*1.0+padded[x+1][y+2]*2.0+padded[x+2][y+2]*1.0)/16.0
            output[x][y] = float(abs(output[x][y]))
    return output

########################## Step 4 Functions #################################################
#
# Consists of:
#   Using the simple threshold method 
#   
####################################################################################


def simpleThreshold(pixel_array, image_width, image_height, threshold):
    for x in range(image_height):
        for y in range(image_width):
            if(pixel_array[x][y] >= threshold):
                pixel_array[x][y] = 1
            else:
                pixel_array[x][y] = 0
    return pixel_array
########################## Step 5 Functions #################################################
#
# Consists of:
#   Using the 5x5 Erosion method
#   Using the 5x5 Dilation method 
#   
####################################################################################

def computeDilation8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    padded = createInitializedGreyscalePixelArray(image_width+4, image_height+4)
    for x in range(image_height):
        for y in range(image_width):
            padded[x+2][y+2] = pixel_array[x][y]
    
    for x in range(2, image_height+2):
        for y in range(2, image_width+2):
            if (padded[x-2][y+2] > 0 or padded[x-2][y+1] > 0 or padded[x-2][y] > 0 or padded[x-2][y-1] > 0 or padded[x-2][y-2] > 0 or
                padded[x-1][y+2] > 0 or padded[x-1][y+1] > 0 or padded[x-1][y] > 0 or padded[x-1][y-1] > 0 or padded[x-1][y-2] > 0 or
                padded[x][y+2] > 0 or padded[x][y+1] > 0 or padded[x][y] > 0 or padded[x][y-1] > 0 or padded[x][y-2] > 0 or
                padded[x+1][y+2] > 0 or padded[x+1][y+1] > 0 or padded[x+1][y] > 0 or padded[x+1][y-1] > 0 or padded[x+1][y-2] > 0 or
                padded[x+2][y+2] > 0 or padded[x+2][y+1] > 0 or padded[x+2][y] > 0 or padded[x+2][y-1] > 0 or padded[x+2][y-2] > 0):
                output[x-2][y-2] = 1
    return output

def computeErosion8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    padded = createInitializedGreyscalePixelArray(image_width+4, image_height+4)
    for x in range(image_height):
        for y in range(image_width):
            padded[x+2][y+2] = pixel_array[x][y]
    
    for x in range(2, image_height+2):
        for y in range(2, image_width+2):
            if (padded[x-2][y+2] > 0 and padded[x-2][y+1] > 0 and padded[x-2][y] > 0 and padded[x-2][y-1] > 0 and padded[x-2][y-2] > 0 and
                padded[x-1][y+2] > 0 and padded[x-1][y+1] > 0 and padded[x-1][y] > 0 and padded[x-1][y-1] > 0 and padded[x-1][y-2] > 0 and
                padded[x][y+2] > 0 and padded[x][y+1] > 0 and padded[x][y] > 0 and padded[x][y-1] > 0 and padded[x][y-2] > 0 and
                padded[x+1][y+2] > 0 and padded[x+1][y+1] > 0 and padded[x+1][y] > 0 and padded[x+1][y-1] > 0 and padded[x+1][y-2] > 0 and
                padded[x+2][y+2] > 0 and padded[x+2][y+1] > 0 and padded[x+2][y] > 0 and padded[x+2][y-1] > 0 and padded[x+2][y-2] > 0):
                output[x-2][y-2] = 1
    return output

########################## Step 6 Functions #################################################
#
# Consists of:
#   Queue class 
#   Using the connected components method 
#   
####################################################################################
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    label = 1
    ccsizes = {}
    visited = set()
    output = [[0 for x in range(image_width)] for y in range(image_height)]
    for x in range(image_height):
        for y in range(image_width):
            if ((pixel_array[x][y] > 0) and (not (x,y) in visited)):
                count = 1
                queue = Queue()
                queue.enqueue((x,y))
                visited.add((x, y))
                while (queue.isEmpty() == False):
                    tuple = queue.dequeue()
                    i = tuple[0]
                    j = tuple[1]
                    output[i][j] = label
                    if (i - 1 >= 0):
                        if ((pixel_array[i-1][j] > 0) and (not (i-1,j) in visited)):
                            count = count + 1
                            visited.add((i-1,j))
                            queue.enqueue((i-1,j))
                    if (i + 1 < image_height):
                        if ((pixel_array[i+1][j] > 0) and (not (i+1,j) in visited)):
                            count = count + 1
                            visited.add((i+1, j))
                            queue.enqueue((i+1,j))
                    if (j - 1 >= 0):
                        if ((pixel_array[i][j-1] > 0) and (not (i,j-1) in visited)):
                            count = count + 1
                            visited.add((i,j-1))
                            queue.enqueue((i,j-1))
                    if (j + 1 < image_width):
                        if ((pixel_array[i][j+1] > 0) and (not (i,j+1) in visited)):
                            count = count + 1
                            visited.add((i,j+1))
                            queue.enqueue((i,j+1))
                ccsizes[label] = count
                label = label + 1
    return (output, ccsizes)

def drawBoundingBox(ccimage, ccsizes,image_width, image_height):
    keyList = list(ccsizes.keys())
    valuesList = list(ccsizes.values())

    #We assume that the biggest component is the barcode
    biggest = 0
    for x in range(len(valuesList)):
        if valuesList[x] > biggest:
            biggest = valuesList[x]
    index = valuesList.index(biggest)

    #Define the initial coordinate of the image
    minX = image_width
    minY = image_height
    maxX = 0
    maxY = 0

    for i in range(image_height):
        for j in range(image_width):
            if ccimage[i][j] == keyList[index]:
                if i < minY:
                    minY = i
                elif i > maxY:
                    maxY = i
                elif j < minX:
                    minX = j
                elif j > maxX:
                    maxX = j
    
    return (minY, maxY, minX, maxX)
    
# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode7"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION here
    px_array = seperateArraysToRGB( px_array_r, px_array_g, px_array_b, image_width, image_height)
    greyscale_pixel_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    stretched_pixel_array = scaleTo0And255AndQuantize(greyscale_pixel_array, image_width, image_height)
    #output_array_vertical = computeVerticalEdgesSobelAbsolute(stretched_pixel_array, image_width, image_height)
    #output_array_horizontal = computeHorizontalEdgesSobelAbsolute(stretched_pixel_array, image_width, image_height)
    #output_array = abs(output_array_horizontal - output_array_vertical)
    output_array = computeStandardDeviationImage5x5(stretched_pixel_array, image_width, image_height)
    output_arrayOne = computeGaussianAveraging3x3RepeatBorder(output_array, image_width, image_height)
    output_arrayTwo = computeGaussianAveraging3x3RepeatBorder(output_arrayOne, image_width, image_height)
    output_arrayThree = computeGaussianAveraging3x3RepeatBorder(output_arrayTwo, image_width, image_height)
    output_arrayFour = computeGaussianAveraging3x3RepeatBorder(output_arrayThree, image_width, image_height)
    output_arrayFinal = computeGaussianAveraging3x3RepeatBorder(output_arrayFour, image_width, image_height)
    threshold_image =  simpleThreshold(output_arrayFinal, image_width, image_height, 26)
    erosion_one = computeErosion8Nbh5x5FlatSE(threshold_image, image_width, image_height)
    erosion_two = computeErosion8Nbh5x5FlatSE(erosion_one, image_width, image_height)
    erosion_three = computeErosion8Nbh5x5FlatSE(erosion_two, image_width, image_height)
    dilation_one = computeDilation8Nbh5x5FlatSE(erosion_three, image_width, image_height)
    dilation_two = computeDilation8Nbh5x5FlatSE(dilation_one, image_width, image_height)
    dilation_three = computeDilation8Nbh5x5FlatSE(dilation_two, image_width, image_height)
    dilation_four = computeDilation8Nbh5x5FlatSE(dilation_three, image_width, image_height)
    (ccimage, ccsizes) = computeConnectedComponentLabeling(dilation_four, image_width, image_height)
    (minY, maxY, minX, maxX) = drawBoundingBox(ccimage, ccsizes,image_width, image_height)

    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm
    bbox_min_x = minX
    bbox_max_x = maxX
    bbox_min_y = minY
    bbox_max_y = maxY

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()