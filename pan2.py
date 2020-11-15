import cv2
import numpy as np
print (cv2.__version__)

#Right
imageRight = cv2.imread('stitch/L1.jpg')
imageRight = cv2.resize(imageRight,(700,500),None,0.1,0.1)
imageRightgray = cv2.cvtColor(imageRight,cv2.COLOR_RGB2GRAY)

#Left
imageLeft = cv2.imread('stitch/L2.jpg')
imageLeft = cv2.resize(imageLeft,(700,500),None,0.1,0.1)
imageLeftGray = cv2.cvtColor(imageLeft,cv2.COLOR_RGB2GRAY)

#using Shift from cv2

sift = cv2.xfeatures2d.SIFT_create()
# find Keypoints

kpr, desr = sift.detectAndCompute(imageRight,None)
kpl, desl = sift.detectAndCompute(imageLeft,None)


# We can use both Flan and Matcher method

cv2.imshow('Image key points Right', cv2.drawKeypoints(imageRight,kpr,None))
cv2.imshow('Image key points Left', cv2.drawKeypoints(imageLeft,kpl,None))

#cv2.waitKey(0)
#cv2.destroyAllWindows()
#FLAN_INDEX_KDTREE = 0;
#Index_Params = dict(algorithm=FLAN_INDEX_KDTREE,trees=5)
#Search_Params = dict(checks=50)
#match = cv2.FlannBasedMatcher(Index_Params,Search_Params)


match = cv2.BFMatcher()
matches = match.knnMatch(desr,desl,k=2)
print(matches)

good=[]
not_good = []
for m,n in matches:
    if(m.distance < 0.75 * n.distance):
        good.append([m])
    else:
        not_good.append(m)



draw_param = dict(matchColor = (0,255,0),flags=2)
img3 =cv2.drawMatchesKnn(imageRight,kpr,imageLeft,kpl,good,None,**draw_param)
#img4 =cv2.drawMatches(imageLeft,kpl,imageRight,kpr,matches,None,**draw_param)
cv2.imshow("Wow", img3)



def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result


# Homography graph Best matching points
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    print("**** Strated the Panaroma Stitching*******")
    src_pts = np.float32([kpr[m[-1].queryIdx].pt for m in good]).reshape(-1,1,2)
    des_pts = np.float32([kpl[m[-1].trainIdx].pt for m in good]).reshape(-1,1,2)
    M,mask  = cv2.findHomography(src_pts,des_pts,cv2.RANSAC,5.0)
    result = warpTwoImages(imageLeft,imageRight, M)
    cv2.imshow("Image stitched", result)

else:
    print("Not Enough Matches : Less that Threshold")



cv2.waitKey(0)
cv2.destroyAllWindows()







