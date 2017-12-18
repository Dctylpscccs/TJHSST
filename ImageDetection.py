#EXTERNAL
import sys
import math
import numpy as np
import time
import cv2
import urllib.request
from json.decoder import NaN
#INTERNAL
from PicfVidEx import vidprocess, picprocess

#basic variables
global width, height, channels
width = -1
height = -1
channels = -1
Green = False#commonly used
#maybe have cache of common imgs like x, y, G, etc.

def setup():
    global Green
    full = np.empty((width, height), dtype=np.uint8)
    full.fill(255)
    none = np.empty((width, height), dtype=np.uint8)
    none.fill(0)
    Green = cv2.merge((none,full,none))#merge channels to produce green edges with alpha channel
    Blue = cv2.merge((full,none,none))
    Red = cv2.merge((full,none,none))
    Yellow = cv2.merge((none,full,full))
    
def G(x, y, sd):
    const2 = 2*sd*sd
    const1 = 1/(math.pi*const2) 
    rad = x*x+y*y
    Ga = const1 * math.exp(-rad/const2)
    return Ga
    
def GaussianB(img, radiusx, radiusy, w):
    kernelK = []
    count = 0.0
    for i in range(-radiusx, 1+radiusx, 1):
        row = []
        for j in range(-radiusy, 1+radiusy, 1):
            GG = G(i, j, w)
            count += GG
            row.append(GG)
        kernelK.append(np.array(row))
    Kernel = np.array(kernelK)/count
    dst = cv2.filter2D(src = img, ddepth = -1, kernel = Kernel, borderType = cv2.BORDER_REPLICATE)#-1, same depth as image, default anchor at each pixel, no delta, 
    return dst
    #for more details on edges: http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html
    
def Sobel(img):#calculate left,right and up,down
    kernelx = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])#/4
    kernely = np.array([np.array([-1, -2, -1]), np.array([0, 0, 0]), np.array([1, 2, 1])])#/4
    dstx = cv2.filter2D(src = img, ddepth = -1, kernel = kernelx, borderType = cv2.BORDER_REPLICATE)#const
    dsty = cv2.filter2D(src = img, ddepth = -1, kernel = kernely, borderType = cv2.BORDER_REPLICATE)
    return dstx, dsty
    
def Laplacian(img):
    Kernel = np.array([np.array([0, 1, 0]), np.array([1, -4, 1]), np.array([0, 1, 0])])#/4
    dst = cv2.filter2D(src = img, ddepth = -1, kernel = Kernel, borderType = cv2.BORDER_REPLICATE)
    return dst 

def blobAnal(done, img):
    neighbors = cv2.dilate(img, np.ones((3,3)))#have all neighbors
    sub = cv2.subtract(neighbors, img)
    inter = cv2.bitwise_and(sub, done)
    done = cv2.subtract(done, inter)#remove viable from done
    img = cv2.add(img, inter)
    if cv2.countNonZero(inter) == 0:#no more viable neighbors
        return img
    else:
        return blobAnal(done, img)        
    
def Bilateral(img):
    BID = cv2.bilateralFilter(img,9,50,10,cv2.BORDER_REPLICATE)#source, neighborhood, sigma intensity, sigma distance, border type (REPLICATE)
    return BID

def EdgeDetection(img):
    img1 = Bilateral(img).astype(np.float32)
    x,y = Sobel(img1)
    X = np.multiply(x,x)
    Y = np.multiply(y,y)
    SUM = np.add(X, Y)
    G = np.sqrt(SUM)
    angle = np.arctan(np.divide(y, x))#going to have some divide by zeros
    threshold = 50
    maxbin = 255*255*2
    ret1, M = cv2.threshold(G, threshold, maxbin, cv2.THRESH_TOZERO)
    G = M
    v1 = math.pi/2#90 deg
    v2 = 0
    v3 = math.pi/4#45
    v4 = -math.pi/4#-45
    rest = math.pi/8
    N = G.copy()
    V1 = np.empty((width,height))
    V1.fill(v1)
    V2 = np.empty((width,height))
    V2.fill(v2)
    V3 = np.empty((width,height))
    V3.fill(v3)
    V4 = np.empty((width,height))
    V4.fill(v4)
    Int1 = np.abs(np.subtract(angle, V1))
    Int2 = np.abs(np.subtract(angle, V2))
    Int3 = np.abs(np.subtract(angle, V3))
    Int4 = np.abs(np.subtract(angle, V4))
    mat1b = np.array([np.array([0, -1, 0]), np.array([0, 1, 0]), np.array([0, 0, 0])])
    mat1c = np.array([np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0])])
    mat2b = np.array([np.array([0, 0, 0]), np.array([-1, 1, 0]), np.array([0, 0, 0])])
    mat2c = np.array([np.array([0, 0, 0]), np.array([0, 1, -1]), np.array([0, 0, 0])])
    mat3b = np.array([np.array([0, 0, -1]), np.array([0, 1, 0]), np.array([0, 0, 0])])
    mat3c = np.array([np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([-1, 0, 0])])
    mat4b = np.array([np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 0])])
    mat4c = np.array([np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, -1])])
    mask1b = cv2.filter2D(N, -1, mat1b, borderType = cv2.BORDER_REPLICATE)
    mask2b = cv2.filter2D(N, -1, mat2b, borderType = cv2.BORDER_REPLICATE)
    mask3b = cv2.filter2D(N, -1, mat3b, borderType = cv2.BORDER_REPLICATE)
    mask4b = cv2.filter2D(N, -1, mat4b, borderType = cv2.BORDER_REPLICATE)
    mask1c = cv2.filter2D(N, -1, mat1c, borderType = cv2.BORDER_REPLICATE)
    mask2c = cv2.filter2D(N, -1, mat2c, borderType = cv2.BORDER_REPLICATE)
    mask3c = cv2.filter2D(N, -1, mat3c, borderType = cv2.BORDER_REPLICATE)
    mask4c = cv2.filter2D(N, -1, mat4c, borderType = cv2.BORDER_REPLICATE)
    cond1 = np.logical_and(Int1 <= rest, np.logical_or(mask1b < 0, mask1c < 0))
    cond2 = np.logical_and(np.logical_or(Int2 <= rest, x==0), np.logical_or(mask2b < 0, mask2c < 0))
    cond3 = np.logical_and(Int3 <= rest, np.logical_or(mask3b < 0, mask3c < 0))
    cond4 = np.logical_and(Int4 <= rest, np.logical_or(mask4b < 0, mask4c < 0))
    result = np.logical_or(np.logical_or(cond1, cond2), np.logical_or(cond3, cond4))
    N[result == True] = 0
    me, sig = cv2.meanStdDev(N)#for now
    minVal = int(max(0, me-sig))#approx under one std
    maxVal = int(min(255, me+sig))#approx above one std
    ret1, init = cv2.threshold(N, maxVal, 255, cv2.THRESH_BINARY)#contains viable pixels, white is above max
    ret2, Hlow = cv2.threshold(N, minVal, 255, cv2.THRESH_BINARY)#white is above min
    init = init.astype(np.uint8)
    Hlow = Hlow.astype(np.uint8)
    Hhi = cv2.bitwise_not(init)
    H = cv2.bitwise_and(Hlow, Hhi)
    H = blobAnal(H, init)
    I0 = np.clip(N,0,255).astype(np.uint8)
    me,sd = cv2.meanStdDev(I0)
    ret3, I = cv2.threshold(I0, thresh = me+1.5*sd, maxval = 255, type = cv2.THRESH_BINARY)#255 or 0
    return I

def CornerDetection(img):
    img1 = img.astype(np.float32)
    img2 = cv2.cornerHarris(img1, 2, 3, .04)
    img3 = cv2.dilate(img2, kernel = np.ones((3,3)).astype(np.float32))
    me = np.mean(img3)
    sd = np.std(img3)
    ret, img4 = cv2.threshold(img3, me, 255, 0)
    img5 = img4.astype(np.uint8)
    ret1, labels, stats, centroids = cv2.connectedComponentsWithStats(img5)#finds centroids
    crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.001)#create criteria
    corn = cv2.cornerSubPix(img1, np.float32(centroids),(5,5),(-1,-1), crit)
    cornW = np.rint(corn)#int format
    img6 = np.zeros((width, height))
    for w in cornW:
        if w[1] > width-1 or w[0] > height-1:continue
        if img5[w[1]][w[0]] != 0:
            img6[w[1]][w[0]] = 255#corners are bold
    img7 = img6.astype(np.uint8)
    return img7

def findStraights(img, imgE):
    p = np.zeros((width,height))#x or column based
    for c in range(height):
        p[0:width,c] = c
    q = np.zeros((width,height))
    for r in range(width):
        q[r,0:height] = r 
    full = np.empty((width,height))
    full.fill(255)
    imgEE = np.divide(imgE, full)
    P = cv2.multiply(p, imgEE) 
    Q = cv2.multiply(q, imgEE) 
    lines = set()
    error = 2.0#+/-
    threshold = min(width,height)/3.5#2.7
    ERROR = np.empty((width,height))
    ERROR.fill(error)
    counter = -np.pi/180#1 degree
    start = np.pi/2
    end = -np.pi/2
    houghspace = np.empty((180, int(math.sqrt(width*width+height*height))))# d < 1000
    theta = start
    it = 0
    
    START = time.time()
    
    while theta > end:
        COS = np.cos(theta)
        SIN = np.sin(theta)
        d = np.add(np.multiply(P,COS), np.multiply(Q,SIN))
        d = np.divide(d,error)
        d = np.rint(d)#now all the d's are approx
        D = list(np.reshape(d, width*height))
        un, freq = np.unique(D, return_counts = True)
        did = d.flatten()
        COC = imgE.flatten()
        ice = np.nonzero(COC)[0]
        unx, freqx = np.unique(did[ice], return_counts = True)
        for i in range(len(unx)):
            houghspace[it][unx[i]] = freqx[i]
        un[freq<threshold] = 0 
        indices = np.nonzero(un)[0]
        for i in indices:
            rho = un[i]*error
            lines.add((rho, theta))
        theta += counter
        it += 1
    maxi = 255.0/np.max(houghspace) 
    maxiF = np.empty((houghspace.shape))
    maxiF.fill(maxi)
    houghspace=np.multiply(houghspace, maxiF)
    houghspace = houghspace.astype(np.uint8)
    cv2.imshow("Cliao", houghspace)
    counter = np.pi/180
    Lines = []
    errt = 7
    errd = min(width,height)/40
    
    print(time.time()-START)
    
    for l in lines:
        (rho1,t1) = l
        done1 = False 
        for i in range(len(Lines)): 
            (rho2,t2,fr) = Lines[i]
            if (abs(t2-t1) < errt*counter and abs(rho2-rho1) < errd):
                done1 = True
                rho2 = (rho2*fr+rho1)/(fr+1)
                t2 = (t2*fr+t1)/(fr+1)
                fr += 1
                Lines[i] = (rho2,t2,fr)
            if (abs(t2+t1) < errt*counter and abs(rho2+rho1) < errd):
                done1 = True
                rho2 = (rho2*fr-rho1)/(fr+1)
                t2 = (t2*fr-t1)/(fr+1)
                fr += 1
                Lines[i] = (rho2,t2,fr)
        if done1 == False:
            Lines.append((rho1,t1,1))
    drawlines = np.zeros((width,height))
    
    print(time.time()-START)
    
    for L in Lines:#accepted:  
        (rho, theta, fr) = L #change back for other method
        a = np.cos(theta) 
        b = np.sin(theta) 
        x0 = a*rho 
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(drawlines, (x1,y1),(x2,y2),255,1)  
        #will have to multiply error back
    cv2.imshow("dl", drawlines)
    return Lines#accepted

def findCircles(img, imgE):
    img1 = Bilateral(img).astype(np.float32)
    x,y = Sobel(img1)
    angle = np.arctan(np.divide(y, x))#going to have some divide by zeros
    e1, e2 = np.nonzero(imgE)#might ignore when x==0
    cv2.imshow("E",imgE)
    mul = np.empty((width,height))
    mul.fill(255/np.pi)
    adda = np.empty((width,height))
    adda.fill(255/np.pi)
    showA = angle.copy() 
    showA = np.nan_to_num(showA).astype(np.float64)
    showA = cv2.add(cv2.multiply(showA, mul),adda)
    cv2.imshow("a",showA.astype(np.uint8))
    COUNT = np.zeros((width,height))
    for i in range(len(e1)):
        count = np.zeros((width,height))
        y0 = e1[i]
        x0 = e2[i]
        thet = angle[y0][x0]
        if x[y0][x0] == 0:
            a = 1
            b = 0
        else:
            b = np.cos(thet)
            a = np.sin(thet)
        x1 = int(x0 + 1000*(b))#the 1000 may need to be changed
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(b))
        y2 = int(y0 - 1000*(a))
        cv2.line(count,(x1,y1),(x2,y2),1,1)#counter
        COUNT = cv2.add(COUNT, count)
    COUNT = GaussianB(COUNT,1,1,1)
    me,std = cv2.meanStdDev(COUNT)#before laplacian
    thres = me + 2*std
    COUNT[COUNT<thres] = 0
    maxmat1 = np.array([np.array([-1,0,0]),np.array([0,1,0]),np.array([0,0,0])])
    maxmat2 = np.array([np.array([0,-1,0]),np.array([0,1,0]),np.array([0,0,0])])
    maxmat3 = np.array([np.array([0,0,-1]),np.array([0,1,0]),np.array([0,0,0])])
    maxmat4 = np.array([np.array([0,0,0]),np.array([-1,1,0]),np.array([0,0,0])])
    maxmat5 = np.array([np.array([0,0,0]),np.array([0,1,-1]),np.array([0,0,0])])
    maxmat6 = np.array([np.array([0,0,0]),np.array([0,1,0]),np.array([-1,0,0])])
    maxmat7 = np.array([np.array([0,0,0]),np.array([0,1,0]),np.array([0,-1,0])])
    maxmat8 = np.array([np.array([0,0,0]),np.array([0,1,0]),np.array([0,0,-1])])
    reduced1 = cv2.filter2D(COUNT, -1, maxmat1, borderType=cv2.BORDER_REPLICATE)
    reduced2 = cv2.filter2D(COUNT, -1, maxmat2, borderType=cv2.BORDER_REPLICATE)
    reduced3 = cv2.filter2D(COUNT, -1, maxmat3, borderType=cv2.BORDER_REPLICATE)
    reduced4 = cv2.filter2D(COUNT, -1, maxmat4, borderType=cv2.BORDER_REPLICATE)
    reduced5 = cv2.filter2D(COUNT, -1, maxmat5, borderType=cv2.BORDER_REPLICATE)
    reduced6 = cv2.filter2D(COUNT, -1, maxmat6, borderType=cv2.BORDER_REPLICATE)
    reduced7 = cv2.filter2D(COUNT, -1, maxmat7, borderType=cv2.BORDER_REPLICATE)
    reduced8 = cv2.filter2D(COUNT, -1, maxmat8, borderType=cv2.BORDER_REPLICATE)
    error = -1
    COUNT[reduced1<error] = 0
    COUNT[reduced2<error] = 0
    COUNT[reduced3<error] = 0
    COUNT[reduced4<error] = 0
    COUNT[reduced5<error] = 0
    COUNT[reduced6<error] = 0
    COUNT[reduced7<error] = 0
    COUNT[reduced8<error] = 0
    centery, centerx = np.nonzero(COUNT)
    #Centers should be maximums among their neighbors: if not so, make 0
    radmin = min(width,height)/20
    radmax = int(min(width, height)/2)
    combs = []
    const = 2*np.pi/1.2#multiply it by needed
    for i in range(len(centerx)):
        ey = e1.copy() 
        ex = e2.copy()
        cx = centerx[i]
        cy = centery[i]
        dist = np.rint(np.sqrt(np.add(np.square(np.subtract(ex,cx)), np.square(np.subtract(ey,cy))))).astype(np.int32)
        Dist = dist[dist<radmax]#take out too high values
        freq = np.bincount(Dist)
        posrad = np.nonzero(freq)[0]
        interest = freq[posrad]
        con = interest#np.add(np.add(minus1, plus1),interest)#sum of edge pixels in viable area
        mul1 = np.zeros((len(posrad)))
        mul1.fill(const)
        cons = np.subtract(con, np.multiply(np.power(posrad, .75),mul1))#bigger radius have more error
        cons[cons < 0] = 0
        indexg = np.nonzero(cons)#indices of good radii
        goodrad = posrad[indexg]#find all the good radii
        for g in goodrad:
            if g <= radmin:continue
            combs.append((cx,cy,g))
    #remove similar tuples
    thresb = min(width,height)/40
    Combs = []
    for c in combs:
        done1 = False
        (cx, cy, cr) = c
        for i in range(len(Combs)):
            (Cx, Cy, Cr, fr) = Combs[i]
            if abs(Cx-cx) < thresb and abs(Cy-cy) < thresb and abs(Cr-cr) < thresb:
                done1 = True
                Combs[i][0] = (Cx*fr+cx)/(fr+1)
                Combs[i][1] = (Cy*fr+cy)/(fr+1)
                Combs[i][2] = (Cr*fr+cr)/(fr+1)
                Combs[i][3] += 1
                break
        if done1 == False:
            Combs.append([cx, cy, cr, 1])
    imgCirc = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)#np.zeros((width,height))
    for C in Combs:
        (xx, yy, r, f) = C
        xx = int(xx)
        yy = int(yy)
        r = int(r)
        cv2.circle(imgCirc, (xx, yy), r, (0,0,255))#red
    imgCirc = imgCirc.astype(np.uint8)
    cv2.imshow("C", imgCirc)
    return Combs

def Overlay(foreground, weight, background):#rgb mat, weight alpha mat, rgb mat
    one = np.ones((width, height))
    alpha = weight/255.0
    beta = np.subtract(one, alpha)
    r, g, b = cv2.split(foreground)
    firstR = np.multiply(r, alpha)
    firstG = np.multiply(g, alpha)
    firstB = np.multiply(b, alpha)
    N = background.shape
    if len(N) == 2:#no channel but one
        r = g = b = background
    else:
        r, g, b = cv2.split(background)
    secondR = np.multiply(r, beta)
    secondG = np.multiply(g, beta)
    secondB = np.multiply(b, beta)
    sumR = np.add(firstR, secondR) 
    sumG = np.add(firstG, secondG) 
    sumB = np.add(firstB, secondB) 
    fini = cv2.merge((sumR, sumG, sumB))
    return fini.astype(np.uint8)

def findLines(use, linex, liney, pix):
    x = pix[0]
    y = pix[1]
    neighbors = {(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)}
    for n in neighbors:
        if n in use:
            use.remove(n)
            linex.append(n[0])
            liney.append(n[1])
            use, linex, liney = findLines(use, linex, liney, n)
    return use, linex, liney

def intersect(line1, line2):
    (R1, thet1) = line1
    (R2, thet2) = line2
    A1 = np.cos(thet1) 
    B1 = np.sin(thet1)
    A2 = np.cos(thet2)
    B2 = np.sin(thet2)
    d = (B2*A1-B1*A2)
    if d == 0:
        return 0
    else:
        x = (B2*R1-B1*R2)/d
        y = (A1*R2-A2*R1)/d
    return int(x), int(y)

def inseq(small, big):
    for i in range(1 + len(big) - len(small)):
        if small == big[i:i+len(small)]:
            return True
    return False

def AnalyzeLines(imgE, imgC, img):##edges, corners
    takeaway = cv2.bitwise_not(cv2.dilate(imgC, np.ones((3,3))))            
    imge = cv2.multiply(imgE, takeaway)
    usex, usey = np.nonzero(imge)
    use = set(zip(usex, usey))
    linesx = []
    linesy = []
    #USE LINES
    imgT = np.zeros((width,height))
    T0 = 20
    thres = int(min(width, height)/T0)#THRESHOLD
    while len(use) != 0:
        u = use.pop()
        use, linex, liney = findLines(use, [u[0]], [u[1]], u)
        if len(linex) >= thres:
            for i in range(len(linex)):
                imgT[linex[i]][liney[i]] = 255
            linesx.append(linex)
            linesy.append(liney)
    imgH = np.zeros((width, height))
    T1 = 3.75#3.9#maybe look at average length of all the lines?
    thres2 = int(min(width, height)/T1)
    img1 = np.add(imgT,cv2.dilate(imgC,np.ones((2,2))))
    img2 = cv2.dilate(img1,np.ones((2,2)))
    imgT = img2.astype(np.uint8)
    cv2.imshow("H",imgH)
    cv2.imshow("T",imgT)
    lines = cv2.HoughLines(imgT,1,np.pi/180,thres2)
    look = []
    for e in lines:
        E = e[0]
        rho = E[0]
        theta = E[1]
        if rho < 0:
            rho *= -1
            theta -= np.pi
        a = np.cos(theta) 
        b = np.sin(theta) 
        x0 = a*rho 
        y0 = b*rho
        look.append((rho, theta))#save theta instead? may save space and some computation
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(imgH,(x1,y1),(x2,y2),(255,255,255),1)
    thresR = .03*min(width, height)
    thresT = 3*np.pi/180#2 BEFORE
    accepted = []
    accepted.append((look[0][0],look[0][1],1))
    for i in range(1, len(look)):
        L = look[i]
        (R, thet) = L
        found = False
        for j in range(len(accepted)):
            (R1, thet1, freq) = accepted[j]
            if (abs(R-R1) < thresR and ((2*np.pi-abs(thet-thet1))%(2*np.pi) < thresT or abs(thet-thet1)%(2*np.pi) < thresT)) or (abs(R+R1) < thresR and ((2*np.pi - abs(thet+thet1))%(2*np.pi) < thresT or abs(thet+thet1)%(2*np.pi) < thresT)):
                accepted[j] = ((R1*freq+R)/(freq+1),(thet1*freq+thet)/(freq+1),freq+1)
                found = True
        if found == False:
            accepted.append((R,thet,1))
    insec = {}
    para = {}
    thro = False
    thro2 = False
    thresxy = .05
    thresp = .01#.04#.01
    for i in range(len(accepted)):
        thro2 = False
        R1, thet1, c = accepted[i]
        A1 = np.cos(thet1)
        B1 = np.sin(thet1)
        if len(para) == 0:
            para[thet1] = set()
            para[thet1].add((R1, thet1))
        else:
            for p in para:
                if abs(thet1-p) < thresp:
                    para[p].add((R1, thet1))
                    thro2 = True
            if thro2 == False:
                para[thet1] = set()
                para[thet1].add((R1,thet1))
        for j in range(i+1, len(accepted)):
            R2, thet2, c2 = accepted[j]
            A2 = np.cos(thet2)
            B2 = np.sin(thet2)
            d = (B2*A1-B1*A2)
            if d == 0:
                x = NaN
                y = NaN
            else:
                x = (B2*R1-B1*R2)/d
                y = (A1*R2-A2*R1)/d
            thro = False
            if len(insec) == 0:
                insec[(x,y)] = set()
                insec[(x,y)].add((R1,thet1))#save lines involved in intersection
                insec[(x,y)].add((R2,thet2))
            else:
                for e in insec:
                    x2,y2 = e
                    if abs((x-x2)/x2) < thresxy and abs((y-y2)/y2) < thresxy:
                        insec[e].add((R1,thet1))
                        insec[e].add((R2,thet2))#add to count
                        thro = True
                if thro == False:
                    insec[(x,y)] = set()
                    insec[(x,y)].add((R1,thet1))
                    insec[(x,y)].add((R2,thet2))
    Insec = {}
    Para = {}
    thro3 = False
    thro4 = False
    thresXY = .05
    thresP = .01
    for i in insec:
        (x0,y0) = i
        k = insec[i]
        thro3 = False
        if len(Insec) == 0:
            Insec[i]=k
            continue
        for I in Insec:
            (x1,y1) = I
            if len(k.intersection(Insec[I])) > 2 and abs((x1-x0)/x0) < thresXY and abs((y1-y0)/y0) < thresXY:
                Insec[I] = Insec[I].union(k)
                thro3 = True 
        if thro3 == False:
            Insec[i]=k
    for p in para: 
        k = para[p]
        thro4 = False 
        if len(Para) == 0: 
            Para[p]=k #the set
            continue 
        for P in Para:
            if len(k.intersection(Para[P])) > 2 and abs(P-p) < thresP:
                Para[P] = Para[P].union(k)
                thro4 = True 
        if thro4 == False:
            Para[p]=k
    thres = 8
    imgK = np.zeros((width,height))
    MSet = []
    c0 = 0
    imgI = np.zeros((width,height))
    color = [(255,255,255), (255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0)]
    for i in Insec:
        for q in Insec[i]:
            c0 += 1
            (R, thet) = q
            a = np.cos(thet) 
            b = np.sin(thet)
            x0 = R*a
            y0 = R*b
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))
            cv2.line(imgI,(x1,y1),(x2,y2),(255,255,255),1)
        if len(Insec[i]) >= thres:
            MSet.append(Insec[i])
            for j in Insec[i]:
                c0 += 1
                (R, thet) = j
                a = np.cos(thet) 
                b = np.sin(thet)
                x0 = R*a
                y0 = R*b
                x1 = int(x0 + 2000*(-b))
                y1 = int(y0 + 2000*(a))
                x2 = int(x0 - 2000*(-b))
                y2 = int(y0 - 2000*(a))
                cv2.line(imgK,(x1,y1),(x2,y2),255,1)
    cv2.imshow("I",imgI)            
    imgP = np.zeros((width,height))
    for p in Para:
        for q in Para[p]:
            (R, thet) = q
            a = np.cos(thet) 
            b = np.sin(thet)
            x0 = R*a
            y0 = R*b
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))
            cv2.line(imgP,(x1,y1),(x2,y2),(255,255,255),1)
        if len(Para[p]) >= thres:
            MSet.append(Para[p])
            for j in Para[p]:
                (R, thet) = j
                a = np.cos(thet) 
                b = np.sin(thet)
                x0 = R*a
                y0 = R*b
                x1 = int(x0 + 2000*(-b))
                y1 = int(y0 + 2000*(a))
                x2 = int(x0 - 2000*(-b))
                y2 = int(y0 - 2000*(a))
                cv2.line(imgK,(x1,y1),(x2,y2),255,1)
    cv2.imshow("P",imgP)
    cv2.imshow("K",imgK)
    #sort M by size, then do this
    fined = []
    pq = []
    for M in MSet:
        ele = (-len(M), M)
        pq.append(ele)
    pq.sort()
    for P in pq:#want no line to be in two sets
        (s, M) = P
        done = False
        for i in range(len(fined)):
            f = fined[i]
            if len(M.intersection(f)) > 6:
                fined[i] = f.union(M)
                done = True
                break
            if len(M.intersection(f)) > 0:
                done = True
                break
        if done == False:
            fined.append(M)
    count = len(fined)
    MSet = fined
    if count >= 2:#two intersection points in diff places
        print("IS CHESSBOARD")
    else:
        return np.zeros((width, height))
    colors = [(0,255,255), (255,255,0), (255,0,255), (0,0,255), (0,255,0), (255,0,0), (255,255,255),(0,255,255), (255,255,0),(255,0,255), (0,0,255), (0,255,0), (255,0,0), (255,255,255)]
    imgN = np.zeros((width,height,3))
    for i in range(len(MSet)):#sort by rho
        N = list(MSet[i])
        N.sort()
        MSet[i] = N
        for n in range(len(N)):
            (R, thet) = N[n]
            a = np.cos(thet) 
            b = np.sin(thet)
            x0 = R*a
            y0 = R*b
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))
            cv2.line(imgN,(x1,y1),(x2,y2),colors[n%7],1)
    cv2.imshow("N", imgN)
    boxes = []
    mean, sd = cv2.meanStdDev(img)
    threslow = mean-sd/2
    threshigh = mean-sd/2
    for i in range(len(MSet[0])-1):
        boxes.append([])
        line0 = MSet[0][i]
        line1 = MSet[0][i+1]
        for j in range(len(MSet[1])-1):
            boxes[i].append(0)
            line2 = MSet[1][j]
            line3 = MSet[1][j+1]
            x0, y0 = intersect(line0, line2)
            x1, y1 = intersect(line0, line3)
            x2, y2 = intersect(line1, line2)
            x3, y3 = intersect(line1, line3)
            P = np.array([[x0,y0], [x1,y1], [x2,y2], [x3,y3]])
            M1 = np.zeros((width, height))
            M1 = cv2.fillConvexPoly(M1, P, 1)
            M2 = cv2.erode(M1, kernel = np.ones((5,5)))
            M3 = np.subtract(M1, M2)
            median = np.median(img[M3!=0])
            if median >= threshigh:
                boxes[i][j] = 1
            elif median < threslow:
                boxes[i][j] = -1
            else:
                boxes[i][j] = 0
    TOP = False
    BOTTOM = False
    LEFT = False
    RIGHT = False
    seq = [-1,1,-1]
    seq2 = [1,-1,1]
    for i in range(len(boxes[0])):
        col = [row[i] for row in boxes]
        pos = col.count(1)
        neg = col.count(-1)
        if pos >= 3 and neg >= 3 and inseq(seq, col) == True:
            TOP = MSet[1][i]
            break
    for i in range(len(boxes[0])-1,-1,-1):
        col = [row[i] for row in boxes]
        pos = col.count(1)
        neg = col.count(-1)
        if pos >= 3 and neg >= 3 and inseq(seq, col) == True:
            BOTTOM = MSet[1][i+1]
            break
    for j in range(len(boxes)):
        row = boxes[j]
        pos = row.count(1)
        neg = row.count(-1)
        if pos >= 3 and neg >= 3 and inseq(seq, row) == True:
            LEFT = MSet[0][j]
            break
    for j in range(len(boxes)-1,-1,-1):
        row = boxes[j]
        pos = row.count(1)
        neg = row.count(-1)
        if pos >= 3 and neg >= 3 and inseq(seq, row) == True:
            RIGHT = MSet[0][j+1]
            break
    X0,Y0 = intersect(TOP, LEFT)
    X1,Y1 = intersect(TOP, RIGHT)
    X2,Y2 = intersect(BOTTOM, LEFT)
    X3,Y3 = intersect(BOTTOM, RIGHT)
    boardcorn = np.zeros((width,height))
    boardcorn[Y0][X0] = 255
    boardcorn[Y1][X1] = 255
    boardcorn[Y2][X2] = 255
    boardcorn[Y3][X3] = 255
    return boardcorn
    
def Chesswork(imgE, imgC, img):
    chesscorn = AnalyzeLines(imgE, imgC, img)#finds corners of board
    return chesscorn

def Prepare(img):#on grayscale
    alpha = 1.2
    #beta = 10
    img1 = img.astype(np.float32)
    mean = np.mean(img1)
    use = np.empty((width,height))
    use.fill(mean)
    use2 = np.empty((width,height))
    use2.fill(alpha)
    img2 = np.add(np.multiply(np.subtract(img1, use), use2), use)
    img3 = np.clip(img2, 0, 255).astype(np.uint8)
    return img3

def Process(img):
    START = time.time()
    Images = []
    imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgG = Prepare(imgg)
    #Average Values
    redavg, greenavg, blueavg, alp = cv2.mean(img)
    print("Red Avg: {}\nGreen Avg: {}\nBlue Avg: {}".format(redavg, greenavg, blueavg))
    #Gaussian Blur
    imgGBlur = GaussianB(imgG, 5, 5, 1)#basic 5x5 weight 1
    #Edge Detection
    imgEdge = EdgeDetection(imgG)
    #Corner Detection
    imgCor = CornerDetection(imgG)
    imgCorV = cv2.dilate(imgCor, np.ones((3,3)))
    #circles = findCircles(imgG,imgEdge)
    #straights = findStraights(imgG,imgEdge)
    ##ANALYSIS
    chesscorn = Chesswork(imgEdge.copy(), imgCor.copy(), imgG.copy())
    #chesscorn = np.zeros((width,height))
    boardcorn = cv2.dilate(chesscorn,np.ones((3,3)))
    ###Formatting
    imgE = Overlay(Green, imgEdge, img)
    imgGE = Overlay(Green, imgEdge, imgG)
    imgGBlurE = Overlay(Green, imgEdge, imgGBlur)
    imgC = Overlay(Green, imgCorV, img)
    imgGC = Overlay(Green, imgCorV, imgG)
    imgGBlurC = Overlay(Green, imgCorV, imgGBlur)
    imgEdgeC = Overlay(Green, imgCorV, imgEdge)
    imgCC = Overlay(Green, boardcorn, img)
    ###add to queue
    Images.append(imgG)
    Images.append(imgGBlur)
    Images.append(imgEdge)
    Images.append(imgCorV)
    Images.append(imgE)
    Images.append(imgGE)
    Images.append(imgGBlurE)
    Images.append(imgC)
    Images.append(imgGC)
    Images.append(imgGBlurC)
    Images.append(imgEdgeC)
    Images.append(imgCC)
    print("Time " + str(time.time()-START))
    return Images
    
def Show(img, Images):
    cv2.namedWindow("Image", flags = 0)
    Img = img
    Title = "Original"
    Type = 1
    while True:
        cv2.imshow("Image", Img)#1 = WINDOW_AUTOSIZE. 0 is normal, 2 is opengl
        #cv2.setWindowTitle("Image", Title)
        key = cv2.waitKey()
        if key == ord('r'):
            Img = img
            Title = "Original"
            Type = 1
        elif key == ord('g'):
            Img = Images[0]
            Title = "Grayscale"
            Type = 2
        elif key == ord('b'):###can continuously blur
            Img = Images[1]
            Title = "Blurred Grayscale"
            Type = 3
        elif key == ord('e'):
            if Type == 1: Img = Images[4]
            elif Type == 2: Img = Images[5]
            elif Type == 3: Img = Images[6]
            Title = "Edge Overlay"
        elif key == ord('c'):
            if Type == 1: Img = Images[7]
            elif Type == 2: Img = Images[8]
            elif Type == 3: Img = Images[9]
            elif Type == 4: Img = Images[10]
            Title = "Corner Overlay"
        elif key == ord('E'):
            Img = Images[2]
            Title = "Edge Only"
            Type = 4
        elif key == ord('C'):
            Img = Images[3]
            Title = "Corner Only"
            Type = 5
        elif key == ord('o'):
            Img = Images[11]
            Title = "Chess Corners"
        elif key == 27: #Escape button pressed
            break
    cv2.destroyAllWindows()
    
def main():
    global width, height, channels
    length = len(sys.argv)
    inname = False
    addi = False
    outname = False
    if length > 1:
        inname = sys.argv[1]
    if length > 2:
        addi = sys.argv[2]
    if length > 3:
        outname = sys.argv[3]
    if inname == "Video":
        START = time.time()
        picstream = vidprocess(addi)#addi would then be a video file
        if len(picstream) == 0:
            sys.exit()
        width, height, channels = picstream[0].shape
        setup()
        chesscorn = np.zeros((width,height))
        cv2.circle(chesscorn, (int(width/2),int(height/2)), int(min(width,height)/3),255, 5)#TESTING
        boardcorn = cv2.dilate(chesscorn,np.ones((3,3)))
        outstream = []
        for p in picstream:
            print(time.time()-START)
            imgCC = Overlay(Green, boardcorn, p)
            #Images = Process(p)
            #chessed = Images[11]#chess corner image
            outstream.append(imgCC)#chessed)
            #Show(p, Images)
        picprocess(outstream, width, height, outname)#save to video
    else:
        if addi != False:
            urllib.request.urlretrieve(addi, inname)
        img = cv2.imread(inname, 1)
        width, height, channels = img.shape
        setup()
        Images = Process(img)
        Show(img, Images)
            
main()
print("END")
