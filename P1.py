import cv2
import numpy as np
import random
from A2 import A2

np.seterr(divide='ignore', invalid='ignore')
sift=cv2.xfeatures2d.SIFT_create()
numIterations = 500
inlierThreshold = 1

a2 = A2()

class P1:
    def main(self):
        a2.box()

        option = input("Enter your input option - 'A' for Rainier Panaroma, 'B' for custom Panaroma and anything else for MelakwaLake Panaroma:  ").strip()
        print("\nDo you want to generate intermediate step images for step 1 and step 2 along with the required images (it will take upto 15 minutes to execute) ?")
        flag = input("Enter 'Y' for yes or 'N' for No:  ").strip()

        if option.upper() == 'A':
            image1 = cv2.imread("project_images/Rainier1.png", 3)
            image2 = cv2.imread("project_images/Rainier2.png", 3)
            image3 = cv2.imread("project_images/Rainier3.png", 3)
            image4 = cv2.imread("project_images/Rainier4.png", 3)
            image5 = cv2.imread("project_images/Rainier5.png", 3)
            image6 = cv2.imread("project_images/Rainier6.png", 3)

            i_1 = image1.copy()
            i_2 = image2.copy()
            i_3 = image3.copy()
            i_4 = image4.copy()
            i_5 = image5.copy()
            i_6 = image6.copy()

            images = [image1, image2, image3, image4, image5, image6]
            image_copies = [i_1, i_2, i_3, i_4, i_5, i_6]

        elif option.upper() == 'B':
            image1 = cv2.imread("project_images/custom/Neigh1.png", 3)
            image2 = cv2.imread("project_images/custom/Neigh4.png", 3)
            image3 = cv2.imread("project_images/custom/Neigh2.png", 3)
            image4 = cv2.imread("project_images/custom/Neigh3.png", 3)
            image5 = cv2.imread("project_images/custom/Neigh5.png", 3)
            image6 = cv2.imread("project_images/custom/Neigh6.png", 3)

            i_1 = image1.copy()
            i_2 = image2.copy()
            i_3 = image3.copy()
            i_4 = image4.copy()
            i_5 = image5.copy()
            i_6 = image6.copy()

            images = [image1, image2, image3, image4, image5, image6]
            image_copies = [i_1, i_2, i_3, i_4, i_5, i_6]

        else:
            image1 = cv2.imread("project_images/MelakwaLake1.png", 3)
            image2 = cv2.imread("project_images/MelakwaLake2.png", 3)
            image3 = cv2.imread("project_images/MelakwaLake3.png", 3)
            image4 = cv2.imread("project_images/MelakwaLake4.png", 3)

            i_1 = image1.copy()
            i_2 = image2.copy()
            i_3 = image3.copy()
            i_4 = image4.copy()

            images = [image1, image2, image3, image4]
            image_copies = [i_1, i_2, i_3, i_4]


        stitched_image = image1.copy()
        stitche_image_copy = image1.copy()


        for i in range(0, len(images)-1):
            print("---- Computing for  Image " + str(i+1) + " and Image " + str(i+2) + " ----")

            print("STEP-1: Computing Harris Corner Detector for the two images")
            if flag.upper() == 'Y':
                i1 = stitched_image.copy()
                i2 = images[i+1].copy()
                a2.main(i1, i2, i+1) #Using Assignment-2 for Step-1 and step-2
            else:
                if i == 0:
                    i1 = stitched_image.copy()
                    i2 = images[i + 1].copy()
                    a2.main(i1, i2, i + 1)  # Using Assignment-2 for Step-1 and step-2

            kp1, des1 = self.compute_descriptor(stitched_image)
            kp2, des2 = self.compute_descriptor(images[i+1])

            print("STEP-2: Matching the interest points of two images")
            matching_result, matches = self.compute_match(stitched_image, kp1, des1, images[i+1], kp2, des2)

            print("STEP-3 : Peforming RANSAC")
            H_f, matches_f = self.RANSAC(matches,kp1,kp2)
            matching_result_f = cv2.drawMatches(stitched_image, kp1, images[i+1], kp2, matches_f, None, flags=2)
            inverseH = self.findInverseHomography(H_f)

            print("STEP-4 : Stitching Image")
            stitched_image = self.stitch(stitche_image_copy, image_copies[i+1], H_f, inverseH)
            stitche_image_copy = stitched_image.copy()

            if i == 0:
                cv2.imwrite("Output/3.png", matching_result_f)
                cv2.imshow("3", matching_result_f)
                cv2.imwrite("Output/4.png", stitched_image)
                cv2.imshow("4", stitched_image)
            else:
                cv2.imwrite("Output/Image_3." + str(i + 1) + ".png", matching_result_f)
                cv2.imwrite("Output/Image_4."+str(i+1) + ".png", stitched_image)

            print("\n")

        cv2.imwrite("Output/Final_Panaroma.png", stitched_image)
        print("\n--------- Now displaying all the Images --------------")
        print("\nPress ENTER KEY by selecting any displayed image to exit the program and close all the displayed images.")

        cv2.imshow("Final_Panaroma", stitched_image)
        key = cv2.waitKey(0)
        if key == 13:  # waiting for enter key to exit
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    def projectCorners(self,corners, H):
        proj_corners = []
        for corner in corners:
            x1 = corner[0]
            y1 = corner[1]
            x2, y2 = self.project(x1,y1,H)
            proj_corners.append([x2,y2])
        return proj_corners


    def computeCorners(self,image):
        corners= []
        corners.append([0,0]) #left top
        corners.append([0,image.shape[0]]) #0,height -leftbottom
        corners.append([image.shape[1],0])  # 0,height -rightTop
        corners.append([image.shape[1], image.shape[0]])  # width,height -rightBottom
        return corners


    def computeSize(self,corners1,proj_corners2):

        x1, y1 = corners1[0]
        xmax, ymax = corners1[3] # x2/xmax = width and y2/ymax = height

        x = []
        y = []
        x.append(x1)
        y.append(y1)
        x.append(xmax)
        y.append(ymax)
        for corner in proj_corners2:
            a1 = corner[0]
            b1 = corner[1]

            x.append(a1)
            y.append(b1)

        xmin = abs(min(x))
        xmax = max(x) + xmin
        ymin = abs(min(y))
        ymax = max(y) + ymin

        return int(xmin), int(ymin), int(xmax), int(ymax)

    def stitch(self,image1, image2, H_f, inverseH):  #STEP-4
        corners1 = self.computeCorners(image1)
        corners2 = self.computeCorners(image2)
        proj_corners2 = self.projectCorners(corners2, inverseH) #project corners of image 2 on image 1 using Inverse H

        print("STEP-4a : Computing size of stitched Image")
        xmin, ymin, xmax, ymax = self.computeSize(corners1,proj_corners2)        #STEP-4a

        stitched_image = np.zeros([ymax, xmax, 3], dtype=np.uint8) #create black image of max size

        #copy image 1 to stitched image                                     #STEP-4b
        print("STEP-4b : copying image 1 to stitched image ")
        for y in range(0,image1.shape[0]): #row-wise
            for x in range(0,image1.shape[1]): #col-wise
                stitched_image[ymin + y][xmin + x] = image1[y][x]

        print("STEP-4c : projecting stitched image points onto image 2")
        for y1 in range(stitched_image.shape[0]): #row-wise                 #STEP-4c
            for x1 in range(stitched_image.shape[1]): #col-wise
                x2, y2 = self.project(x1-xmin,y1-ymin,H_f) #project the point on image 2

                #check if the projected point lies within the boundary of image 2
                if x2 > corners2[0][0] and x2 < corners2[3][0] and y2 > corners2[0][1] and y2 < corners2[3][1]:
                    stitched_image[y1][x1] = cv2.getRectSubPix(image2, (1, 1), (x2, y2))

        return stitched_image


    def findInverseHomography(self,H_f):
        return np.linalg.inv(H_f)


    def project(self,x1,y1,H):
        try:
            a = np.array([[x1], [y1], [1]])
            u, v, w = H.dot(a)
            x2 = u/w
            y2 = v/w
            return x2, y2
        except:
            return x2, y2


    def computeInlierCount(self,H, matches,kp1,kp2):
        inliers = []
        source = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        destination = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        for i in range(0, len(source)):
            x , y = source[i][0]
            x1, y1 = self.project(x,y,H) #projected  x,y from plane 1(source) into  plane 2(destination)
            x2,  y2 = destination[i][0]
            dist = ((x1-x2)**2 + (y1-y2)**2)**(1/2)
            if dist < inlierThreshold:
                inliers.append(i)
        return inliers


    def findHomography(self,random_matches,kp1,kp2):
        source = np.float32([kp1[m.queryIdx].pt for m in random_matches]).reshape(-1, 1, 2)
        destination = np.float32([kp2[m.trainIdx].pt for m in random_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(source, destination, 0)
        return H

    def getRandomPoints(self,matches):
        random_matches = []
        randomlist = random.sample(range(0, len(matches)), 4)

        for j in randomlist:
            random_matches.append(matches[j])
        return random_matches

    def computeRefinedHomography(self,best_H,finalInliers,matches,kp1,kp2):
        matches_f=[]
        for i in finalInliers:
            matches_f.append(matches[i])

        H_f = self.findHomography(matches_f,kp1,kp2) #refined Homography
        return H_f, matches_f


    def RANSAC(self,matches,kp1,kp2):
        best_H  =  []
        count = 0
        finalInliers = []
        for i in range(0,numIterations):
            random_matches = self.getRandomPoints(matches)
            H = self.findHomography(random_matches,kp1,kp2)
            inliers = self.computeInlierCount(H, matches,kp1,kp2)
            if len(inliers) > count:
                count = len(inliers)
                best_H = H # best Homography is found
                finalInliers = inliers

        H_f, matches_f = self.computeRefinedHomography(best_H, finalInliers, matches, kp1, kp2)
        return H_f,matches_f


    def box(self):
        box = cv2.imread("project_images/Boxes.png", 3)
        kp, desc = self.compute_descriptor(box)
        box_x = cv2.drawKeypoints(box, kp, box, color=(0, 0, 255))
        cv2.imwrite('Output/1a.png', box_x)
        cv2.imshow("1a", box_x)

    def compute_descriptor(self,image):
        key_points, descriptor = sift.detectAndCompute(image, None)
        return key_points, descriptor

    def compute_match(self,image1, kp1, des1,image2, kp2, des2):
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)  # NORM_L1 is used for descriptor
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matching_result = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=2)

        return matching_result, matches


if __name__ == "__main__":
    P1().main()