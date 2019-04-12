# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:38:07 2019

@author: Sivasis Jena
"""

import cv2
import numpy as np
import math

def GaussianPyramid(img,level):
    g=img.copy()
    gp=[g]
    for i in range(level):
        g=cv2.pyrDown(g)
        gp.append(g)
    return gp



def LaplacianPyramid(img,level):
    l=img.copy()
    gp=GaussianPyramid(img,level)
    lp=[gp[level]]
    for i in range(level,0,-1):
        size=(gp[i-1].shape[1],gp[i-1].shape[0])
        ge=cv2.pyrUp(gp[i],dstsize=size)
        l=cv2.subtract(gp[i-1],ge)
        lp.append(l)
    lp.reverse()
    return lp

"""
def LaplacianPyramid(img,level):
    gaussPyr=GaussianPyramid(img,level)
    retlst = []
    layers = len(gaussPyr)
    count = 0
    for i in range(layers):
        if i < layers-1:
            r,c = gaussPyr[i].shape
            retlst.append(gaussPyr[i] - cv2.pyrUp(gaussPyr[i+1])[:r,:c])
    retlst.append(gaussPyr[-1])
    return retlst
 """
"""
def PyramidReconstruct(pyramid):
    level=len(pyramid)
    pyr=pyramid[0]
    for i in range(1,level):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        pyr=cv2.pyrUp(pyr,dstsize=size)
        pyr=cv2.add(pyr,pyramid[i])
    return pyr

"""
def PyramidReconstruct(lapl_pyr):
  output = None
  output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
  for i in range(len(lapl_pyr)-1,0,-1):
    lap = cv2.pyrUp(lapl_pyr[i])
    lapb = lapl_pyr[i-1]
    if lap.shape[0] > lapb.shape[0]:
      lap = np.delete(lap,(-1),axis=0)
    if lap.shape[1] > lapb.shape[1]:
      lap = np.delete(lap,(-1),axis=1)
    tmp = lap + lapb
    lapl_pyr.pop()
    lapl_pyr.pop()
    lapl_pyr.append(tmp)
    output = tmp
  return output


def Fusion(w1,w2,img1,img2):
    level=5
    weight1=GaussianPyramid(w1,level)
    weight2=GaussianPyramid(w2,level)
    b1,g1,r1=cv2.split(img1)
    b_pyr1=LaplacianPyramid(b1,level)
    g_pyr1=LaplacianPyramid(g1,level)
    r_pyr1=LaplacianPyramid(r1,level)
    b2,g2,r2=cv2.split(img2)
    b_pyr2=LaplacianPyramid(b2,level)
    g_pyr2=LaplacianPyramid(g2,level)
    r_pyr2=LaplacianPyramid(r2,level)
    b_pyr=[]
    g_pyr=[]
    r_pyr=[]
    for i in range(level):
        b_pyr.append(cv2.add(cv2.multiply(weight1[i],b_pyr1[i]),cv2.multiply(weight2[i],b_pyr2[i])))
        g_pyr.append(cv2.add(cv2.multiply(weight1[i],g_pyr1[i]),cv2.multiply(weight2[i],g_pyr2[i])))
        r_pyr.append(cv2.add(cv2.multiply(weight1[i],r_pyr1[i]),cv2.multiply(weight2[i],r_pyr2[i])))
    b_channel=PyramidReconstruct(b_pyr)
    g_channel=PyramidReconstruct(g_pyr)
    r_channel=PyramidReconstruct(r_pyr)
    out_img=cv2.merge((b_channel,g_channel,r_channel))
    return out_img

def Exposedness(img):
    sigma=0.25
    average=0.5
    row=img.shape[0]
    col=img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    res=np.zeros((row,col), np.float32)
    for i in range(row):
        for j in range(col):
            res[i,j]=math.exp(-1.0*math.pow(gray[i,j]-average,2.0)/(2*math.pow(sigma,2.0)))
    res=(res*255)
    res = cv2.convertScaleAbs(res)
    return res       
    



def main():
    #imgpath="F:\\cs\\Data Science\\computer vision\\standard_test_images\\lena_color_512.tif"
    #img=cv2.imread(imgpath)
    #cv2.imshow("Lena",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #imgpath="F:\\cs\\Data Science\\computer vision\\Underwater_images\\3 wall divers.jpg"
    #img=cv2.imread(imgpath)
    """
    w1=cv2.imread("F:\\cs\\Data Science\\computer vision\\README-images\\w1-4.png")
    w1 = cv2.resize(w1,(512,512),interpolation=cv2.INTER_AREA)
    w2=cv2.imread("F:\\cs\\Data Science\\computer vision\\README-images\\w2-4.png")
    w2 = cv2.resize(w2,(512,512),interpolation=cv2.INTER_AREA)
    i1=cv2.imread("F:\\cs\\Data Science\\computer vision\\README-images\\org-4.png")
    i1 = cv2.resize(i1,(512,512),interpolation=cv2.INTER_AREA)
    i2=cv2.imread("F:\\cs\\Data Science\\computer vision\\README-images\\org-4.png")
    i2 = cv2.resize(i2,(512,512),interpolation=cv2.INTER_AREA)
    cv2.imshow("",Fusion(w1,w2,i1,i2))
    """
    #for i in range(3):
       # print(GaussianPyramid(img,5)[i].shape)
       # print(LaplacianPyramid(gray,5)[i].shape)
    
    i1=cv2.imread("F:\\cs\\Data Science\\computer vision\\Underwater_images\\Anthias and Sarah.jpg")
    img1 = cv2.resize(i1,(512,512),interpolation=cv2.INTER_AREA)
    i2=cv2.imread("F:\\cs\\Data Science\\computer vision\\Underwater_images\\3 wall divers.jpg")
    img2=  cv2.resize(i2,(512,512),interpolation=cv2.INTER_AREA)
    i3=cv2.imread("F:\\cs\\Data Science\\computer vision\\Underwater_images\\Yellowtail Snapper.jpg")
    img3=  cv2.resize(i3,(512,512),interpolation=cv2.INTER_AREA)
    
    i4=cv2.imread("F:\\cs\\Data Science\\computer vision\\Underwater_images\\sarah and the cuttlefish.jpg")
    img4 = cv2.resize(i4,(512,512),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    """
    for i in range(6):
        cv2.imshow("",LaplacianPyramid(img1,5)[1])
        cv2.imwrite("F:\\cs\\Data Science\\computer vision\\Underwater_images\\lap.jpg",img1)
    for i in range(6):
        cv2.imshow("",GaussianPyramid(img1,5)[i])
        """
    cv2.imshow("",Fusion(gray,gray,img4,img4))
    #cow("",Fusion(gray,gray,img2,img4)
    """
    img2=cv2.imread("F:\\cs\\Data Science\\computer vision\\README-images\\org-2.png")
    w1-img2=cv2.imread("F:\\cs\\Data Science\\computer vision\\README-images\\w1-2.png")
    w2-img2=cv2.imread("F:\\cs\\Data Science\\computer vision\\README-images\\w2-2.png")
    cv2.imshow("",Fusion(w1-img2,w2-img2,img2,img2))
    """
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
if __name__=="__main__":
    main()
    
        
    
    
    
    



        
    