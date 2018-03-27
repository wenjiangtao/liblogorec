# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys,os
import time
import glob
cv2.ocl.setUseOpenCL(False)

USE_ORB = False
if USE_ORB:
    sift = cv2.ORB_create()
else:
    sift = cv2.xfeatures2d.SIFT_create()
RC = 0.75
DSIZE = (240,240)
def test(img1f,img2, mth = 10):
    des1 = img1f[1]
    print des1.shape
    kp1 = img1f[0]
    t = time.time()
    if img2.shape[0] > img2.shape[1]:
        img2 = img2[480-240:480+240,360-240:360+240]
    else:
        img2 = img2[360-240:360+240,480-240:480+240]
    img2 = cv2.resize(img2,DSIZE,interpolation=cv2.INTER_AREA)
    kp2, des2 = sift.detectAndCompute(img2,None)
    t1 = time.time()-t
    #print 'detect', t1
    t = time.time()
    # FLANN parameters
    
    if USE_ORB:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
        matches = bf.match(des1,des2)
    else:
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks=50)   # or pass empty dictionary
        # flann = cv2.FlannBasedMatcher(index_params,search_params)
        # matches = flann.knnMatch(des1,des2,k=2)

        bf = cv2.BFMatcher()#crossCheck = True)
        matches = bf.knnMatch(des1,des2, k=2)


    #print 'matches...',len(matches)
    if USE_ORB:
        good = matches
    else:
        # Apply ratio test
        pset1 = set()
        pset2 = set()
        good = []
        for m,n in matches:
            if m.distance < RC*n.distance:
                if m.trainIdx not in pset1 and m.queryIdx not in pset2:
                    pset1.add(m.trainIdx)
                    pset2.add(m.queryIdx)
                    good.append(m)

    t2 = time.time()-t
    #print 'match',t2
    ll = len(good)
    print 'all time',t1+t2,ll
    #print 'good',len(good)
    if ll >= mth:
        if True:
            t3 = time.time()
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            ll = np.count_nonzero(mask)
            print 'homo time',time.time()-t3
            if mth == 10:
                rth = 7
            else:
                rth = mth
            if ll >= rth:
                print 'pass homo with',ll
                return True, ll
            else:
                print 'fail homo with',ll
                return False,ll
        else:
            if ll >= mth:
                return True,ll
    else:
        return False,ll
def test2(img1f,img2f, mth = 10):
    des1 = img1f[1]
    kp1 = img1f[0]
    des2 = img2f[1]
    kp2 = img2f[0]

    t = time.time()

    
    if USE_ORB:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
        matches = bf.match(des1,des2)
    else:
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks=50)   # or pass empty dictionary
        # flann = cv2.FlannBasedMatcher(index_params,search_params)
        # matches = flann.knnMatch(des1,des2,k=2)

        bf = cv2.BFMatcher()#crossCheck = True)
        matches = bf.knnMatch(des1,des2, k=2)

    #print 'matches...',len(matches)
    if USE_ORB:
        good = matches
    else:
        # Apply ratio test
        pset1 = set()
        pset2 = set()
        good = []
        for m,n in matches:
            if m.distance < RC*n.distance:
                if m.trainIdx not in pset1 and m.queryIdx not in pset2:
                    pset1.add(m.trainIdx)
                    pset2.add(m.queryIdx)
                    good.append(m)

    ll = len(good)
    #print 'good',len(good)
    if ll >= mth:
        if True:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            ll = np.count_nonzero(mask)
            if mth == 10:
                rth = 7
            else:
                rth = mth
            if ll >= rth:
                #print 'pass homo'
                return True, ll
            else:
                #print 'fail homo'
                return False,ll
        else:
            if ll >= mth:
                return True,ll
    else:
        return False,ll

dl = glob.glob(sys.argv[1]+'/*')
mlist = {}
thrs = {}
desd = {}
for d in dl:
    fl = glob.glob(d+'/*.jpg') + glob.glob(d+'/*.JPG')
    tpls = glob.glob(d+'/template/*.jpg') + glob.glob(d+'/template/*.png') + glob.glob(d+'/template/*.jpeg')
    tpl = tpls[0]
    print 'matching',tpl,'with',fl
    mlist[tpl] = fl
tpd = 0
tpu = 0
tnd = 0
tnu = 0
OF = open('okr_nt_240.txt','w')
for tpl in mlist:
    print 'testing',tpl
    mm = 10000
    img1 = cv2.imread(tpl,0) # queryImage
    c = 0
    fl = mlist[tpl]
    kp1, des1 = sift.detectAndCompute(img1,None)
    img1r = 255-img1
    kp1r, des1r = sift.detectAndCompute(img1r,None)
    for im in fl:
        img2 = cv2.imread(im,0) # trainImage
        if img2.shape[0] > img2.shape[1]:
            img2 = cv2.resize(img2,(720,960),interpolation = cv2.INTER_AREA)
        else:
            img2 = cv2.resize(img2,(960,720),interpolation = cv2.INTER_AREA)
        r,l = test((kp1,des1),img2)
        r1,l1 = test((kp1r,des1r),img2)
        ll = max(l,l1)
        if r or r1:
            c += 1
            mm = min(mm,ll)
    print>>OF, tpl,'true pos',c,len(fl),'thresh',mm
    print tpl,'true pos',c,len(fl),'thresh',mm
    thrs[tpl] = mm
    desd[tpl] = ((kp1,des1),(kp1r,des1r))
    sys.stdout.flush()

    
cc = 0
cf = 0
cg = 0
at = 0
for tp2 in mlist:
    fl = mlist[tp2]
    cc += len(fl)
    for im in fl:
        img2 = cv2.imread(im,0) # trainImage
        if img2.shape[0] > img2.shape[1]:
            img2 = cv2.resize(img2,(720,960),interpolation = cv2.INTER_AREA)
        else:
            img2 = cv2.resize(img2,(960,720),interpolation = cv2.INTER_AREA)
        ttt = time.time()
        if img2.shape[0] > img2.shape[1]:
            img2 = img2[480-240:480+240,360-240:360+240]
        else:
            img2 = img2[360-240:360+240,480-240:480+240]
        img2 = cv2.resize(img2,DSIZE,interpolation=cv2.INTER_AREA)
        blv = cv2.Laplacian(img2, cv2.CV_32F).var()
        if blv < 100:
            bfg = True
        else:
            bfg = False
        print im,blv,bfg
        bfg = False
        if bfg:
            cf += 1
            continue
        kp2, des2 = sift.detectAndCompute(img2,None)
        goodS = 0
        for tpl in mlist:
            d1 = desd[tpl][0]
            d2 = desd[tpl][1]
            r,l = test2(d1,(kp2,des2),thrs[tpl])
            r1,l1 = test2(d2,(kp2,des2),thrs[tpl])
            ll = max(l,l1)
            if (r or r1):
                bb = np.float32(ll)/thrs[tpl]
                print im,'match',tpl,'with',bb
                if bb > goodS:
                    goodM = tpl
                    goodS = bb
        if goodS > 0:
            if goodM == tp2:
                cg += 1
                print im,'final match',goodM,'correct'
            else:
                print im,'final match',goodM,'wrong! should be',tp2
                print>>OF, im,'final match',goodM,'wrong! should be',tp2
        else:
            print im,'match no! shoud be',tp2
            print>>OF, im,'match no! shoud be',tp2
            cf += 1
        costt = time.time()-ttt
        print 'match time',costt
        at += costt
print>>OF,'true pos',cg,cc,np.float32(cg)/cc
print 'true pos',cg,cc,np.float32(cg)/cc
print>>OF,'no match',cf,cc,np.float32(cf)/cc
print 'no match',cf,cc,np.float32(cf)/cc
print>>OF,'wrong match',cc-cf-cg,cc,np.float32(cc-cf-cg)/cc
print'wrong match',cc-cf-cg,cc,np.float32(cc-cf-cg)/cc
print >>OF,'avg time',at/cc
print 'avg time',at/cc

OF.close()


