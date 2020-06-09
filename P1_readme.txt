*****README FILE*****

PREREQUISITES:-

Programming Language:- Python3

Libraries/Packages:-
	opencv-contrib-python
	opencv-contrib-python-nonfree
	numpy
	math


REQUIRED INPUT/OUTPUT FILES/FOLDERS:-
1. For Rainier Panaroma(CREDIT-1): Have all the 6 images stored in folder project_images : Rainier1.png, Rainier2.png, Rainier3.png, Rainier4.png, Rainier5.png, Rainier6.png
2. For Custom Panaroma(CREDIT-2): Have all the 6 images stored at path project_images/custom : Neigh1.png, Neigh2.png, Neigh3.png, Neigh4.png, Neigh5.png, Neigh6.png
3. For MelakwaLake Panaroma: Have all the 4 images stored in folder project_images : MelakwaLake1.png, MelakwaLake2.png, MelakwaLake3.png, MelakwaLake4.png
4. Have an empty Output folder in the root project folder to enable writing all the output images.
5. A2.py for my own Harris corner detection implementation. (STEP-1 and STEP-2)

RUN FILE:-
1. Put all the required files in the project_images folder.
2. Run P1.py
3. Choose the Panaroma images:
   For Rainier Panaroma(CREDIT-1): enter "A" as the input.
   For Custom Panaroma(CREDIT-2): enter "B" as the input.
   For MelakwaLake Panaroma(CREDIT-2): enter "C" or anything else as the input.
4. Choose if you want to generate intermetiate images for step-1 and step-2(using my own Harris corner detection implementation.)
   For "Yes": enter "Y" (it takes upto 15 minutes to execute)
   For "No": enter "N" (It will generate all the "Required Images" and the "Intermediate Step Images" for Step-3 and step-4 of assignment)
6. Required Output images(1a.png, 1b.png, 1c.png, 2.png, 3.png , 4.png and Final_Panaroma.png) are displayed.
7. Press ENTER KEY by selecting any displayed image to exit the program and close all the displayed images.
8. All the required output images from Step-6 + intermediate step images for the Final_Panaroma are then written in the Output folder.
 Note:- (Make sure there is an empty folder named "Output" present in project's root folder in order to write the output images)


 To execute Panaroma stitching for new images:-
 1. Change Line (58-69): Read input images one after other in the serial order into variable image1,image2,image3 and so on.
 2. Make copies of the input images in variables i_1, i_2, i_3 and so on. For eg: i_1 = image1.copy()
 (Note: The no. of input image and image copy should be same. For eg: copy of image1 is i_1)
 3. Add all the input images to the List images (Line 68).
 4. Add all the input image copies to the List image_copies (Line 69).
 (Note: Copies and original input images should be added in the same order in both the above lists.)


OUTPUT:-
1. Required Images:
Image 1a.png : Harris response on "Boxes.png" using my own Harris corner detection implementation.
Image 1b.png : Harris corner detector for input image-1 of the Panaroma using my own Harris corner detection implementation.
Image 1c.png : Harris corner detector for input image-2 of the Panaroma using my own Harris corner detection implementation.
Image 2.png : Found matches for input image-1 and image-2 of the Panaroma using my own Harris corner detection implementation.
Image 3.png : Final matches for input image-1 and image-2 of the Panaroma after performing RANSAC using opencv sift detector and descriptor.
Image 4.png : Stitched image with input image-1 and image-2 of the Panaroma using opencv sift descriptor.
Image Final_Panaroma.png : Final panaroma created from all the input images using opencv sift descriptor.

2. Intermediate Step Images(These images are written in the Output folder and are not displayed):
For every subsequent input image (assume variable i), following images are written-
 Image_1b.i.png - Result similar to Image 1b.png with next input image.
 Image_1c.i.png - Result similar to Image 1c.png with next input image.
 Image_2.i.png - Result similar to Image 2.png with next input image.
 Image_3.i.png - Result similar to Image 3.png with next input image.
 Image_4.i.png - Result similar to Image 4.png with next input image.


IMPORTANT GLOBAL VARIABLES:-
1. numIterations - Number of iterations for RANSAC algorithm. (Line 8)
2. inlierThreshold - Threshold value used to consider a match as an inlier or outlier in RANSAC. (Line 9)


FUNCTIONALITY IMPLEMENTED:-

1. Function main() of P1.py: a. Call Function box() from module A2.py - to compute Harris response on "Boxes.png" using my own Harris corner detection implementation.
   b. It is the first function executed- take inputs and perform a for-loop for all the input images.
   c. For every set of images, it calls main() in module A2.py to detect keypoints and find matches.
   d. Module A2.py is my own implementation of Harris Corner detector and is used for generating step-1 images(1a.png ,1b.png ,1c.png) and step-2 images(2.png)
   e. Function main() of module P1.py will then use opencv sift descriptor to perform Step-3 and Step-4 of the assignment.
   f. In general, it creates a stitched image for first two images and then for every next input image, it repeat all the previous steps with the result of previous set of images.

2. Function compute_descriptor(): a. It takes input_image as a parameter, compute keypoints and descriptor of the input_image using opencv sift descriptor.
   b. Return keypoints and descriptor of the input_image back to the main() of P1.py .

3. Function compute_match(): a. Take keypoints and descriptor of both input images, compute matches of the images using cv2.BFMatcher and return matches to main().

4. Function RANSAC(): a. Peform RANSAC algorithm by using keypoints and matches of the input images.
   b. For number of iterations performs all the next steps: Call getRandomPoints() - to choose random 4 points from the matches.
   c. Call findHomography() - to compute homography from random chosen points using cv2.findHomography .
   d. Call computeInlierCount() - to find the count of inliers for that homography found in step-c .
   e. Find the Homography with maximum number of inliers.
   f. Call computeRefinedHomography() - to compute Final or the best homography using maximum no. of inliers.
   g. Return final Homography and final matches to main().

5. Function findInverseHomography(): a. Compute Inverse Homography for the homography computed from RANSAC.

6. Function project(): a. Compute projected x2 and y2 co-ordinates using input x1, y1 and homography.

7. Function stitch(): a. Take input images, Homography and Inverse Homography computed above as parameters.
   b. Call computeCorners() - to compute corners of both the input images.
   c. Call projectCorners() - to project corners of image 2 on image 1 using Inverse Homography using Function project().
   d. Call computeSize() - to compute the size of the stitched_image using the corners of image1 and projected corners of image2 on image1.
   e. Create a black image by using xmin , xmax, ymin and ymax computed from step-d.
   f. Copy image 1 to stitched image pixel by pixel.
   g. Project stitched image points onto image 2 using Homography.
   h. Check if the projected point lies within the boundary of image 2, use cv2.getRectSubPix() to copy the point on stitched image.


REFERENCES:-
1. https://answers.opencv.org/question/199318/how-to-use-sift-in-python/?sort=votes
2. https://stackoverflow.com/questions/39940766/bfmatcher-match-in-opencv-throwing-error
3. https://stackoverflow.com/questions/34711985/normalization-in-harris-corner-detector-in-opencv-c
4. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html 
5. https://www.programcreek.com/python/example/89367/cv2.findHomography
6. https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
7. https://www.reduceimages.com/
8. https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html 




 

