import cv2
import numpy as np


#================ Q2 ============
im1 = cv2.imread("output/img1.tiff") # load image in BGR format

im1swapped = im1.copy()
im1swapped[:,:,2], im1swapped[:,:,1] = im1swapped[:,:,1].copy(), im1swapped[:,:,2].copy()

im1_green = im1[:,:,1]
im1_red = im1[:,:,2]

cv2.imwrite('output/swapped_img1.tiff', im1swapped)
cv2.imwrite('output/green_img1.tiff', im1_green)
cv2.imwrite('output/red_img1.tiff', im1_red)
# In this case the red image works best because there is more red in the picture
# It can be supposed that a CV algo would work best on the red because there is more of it

#================ Q3 ====================
imSwapped = im1_green.copy()
cX, cY = np.array(im1_red.shape) // 2
N = 50
imSwapped[cX - N : cX + N, cY - N : cY + N] = im1_red[cX - N : cX + N, cY - N : cY + N]
cv2.imwrite("output/imSwapped.tiff", imSwapped)

#============== Q4 ==============
meanGreen = im1_green.mean()
stdDev = im1_green.std()
# print("Mean of green values is: ", meanGreen, "-- This was found using numpy mean() function")
# print("Std dev is ", stdDev, "which was found using numpy std() function")
n = 90
shift_kernel = np.zeros((n,n), np.float32)
shift_kernel[n//2, 1] = 1

# print(shift_kernel)
imshifted = cv2.filter2D(im1_green, -1, shift_kernel)
cv2.imwrite("output/shifted_im1.tiff", imshifted)

#============== Q5 ================
# map image of arbitary range to float [0,1]
def normalize_im(img):
	img += img.min()
	img /= img.max()
	return img

imsizeX, imsizeY = im1_green.shape
gaussian_kernel = cv2.getGaussianKernel(3, 1)
noise_gaussian = np.random.normal(0, 12, im1_green.shape)
I = normalize_im(im1_green + noise_gaussian)
cv2.imshow("Gaussian Noise, normalized", I)

def truncate_img(img):
	return (np.vectorize(lambda x: max(max(0, x), min(x, 255)))(img)).astype(np.uint8)
A = truncate_img(im1_green + noise_gaussian)
# print(A[0])
cv2.imshow("Green, truncated", A)
cv2.imshow("Red, truncated", truncate_img(im1_red + noise_gaussian))

cv2.waitKey(0)
cv2.destroyAllWindows() 
