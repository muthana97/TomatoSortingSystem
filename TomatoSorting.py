#Importing the libraries 
import numpy as np
import cv2
import time


#broker_address = "mqtt.eclipse.org"
#port= 1883
#try:
 #   print("data uploaded to server")
#    client = mqtt.Client()
 #   client.on_subscribe = on_subscribe
#    client.on_unsubscribe = on_unsubscribe
#    client.on_connect = on_connect
#    client.on_message = on_message
 #   client.connect(broker_address,port)
 ##   time.sleep(1)
#except: 
 #   print("fail")
 #   pass

#Btopic = "/data/Bad"
#Gtopic = "/data/Good" 
#Ttopic = "/data/Total"
#print("initializing WebCam")
#Load the video to be operated on
#cap = cv2.VideoCapture(r'C:\Users\HP\Documents\APU FINAL YEAR\GDP CODELAB\M1.mp4')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#cap = cv2.VideoCapture(0)
ret, frame = cap.read()
#print(frame.shape)
global lower

global tol
tol=18 #tolerence

global Ggood
Ggood = 0 #Total number of tomatoes
#client.publish(Gtopic,Ggood)
global Xprevious #used for counting tomatos
try:
    Xprevious= int(frame.shape[1])
except:
    pass

global sensitivity
sensitivity= 400 #any damage over 400 is considered bad

global Bbad
Bbad=0
#client.publish(Btopic,Bbad)

global Ttotal
#Ttotal = Bbad + Ggood
#client.publish(Ttopic,Ttotal)

while(True):
    #print("while started")
    #####################################
    ##Operations on the frame come here##
    #####################################
    try:
        
        ret, frame = cap.read()
        
        ###Frame Resize
        n_widt  = int(frame.shape[1] * 0.3)
        n_heigh = int(frame.shape[0] * 0.3)
        n_width  = int(frame.shape[0] * 0.3)
        n_height = int(frame.shape[1] * 0.3)
        dsize = (n_widt, n_heigh)
        frame   = cv2.resize(frame, dsize, interpolation = cv2.INTER_AREA)
        frame1  = cv2.resize(frame, dsize, interpolation = cv2.INTER_AREA)
        ###conversion to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
       
        ###Thresholding and binarization
        ret,thresh = cv2.threshold(gray,155,255,cv2.THRESH_BINARY_INV)
        
        ###Region of Detection 
        Ytop=int(frame.shape[1]*0.5)
        Ybottom=int((frame.shape[1]*0.7))
        xb=5
        ROI=(thresh[xb:n_width-xb,Ytop:Ybottom])
        
        #Draw ROI-1
        cv2.line(frame,(Ybottom, xb),(Ytop, xb), (0, 128, 255), 2)
        cv2.line(frame,(Ybottom, n_width-xb),(Ytop, n_width-xb), (0, 128, 255), 2)
        cv2.line(frame,(Ybottom, xb),(Ybottom, n_width-xb), (0, 0, 255), 2)
        cv2.line(frame,(Ytop ,xb),(Ytop, n_width-xb), (0, 128, 255), 2)        
        
        
        
        ###Edge detection
        v = np.median(gray)
        sigma = 0.33
        #---- Apply automatic Canny edge detection using the computed median----
        lower = int(max(0, (1.0 - sigma) * v))
        #upper = int(min(255, (1.0 + sigma) * v))
        #edged = cv2.Canny(thresh, lower, upper)
        
        ##Morphological operations
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(thresh,kernel,iterations = 3)
        erosion = cv2.erode(dilation,kernel,iterations = 2)
        
        ##Circle Detection
        circles = cv2.HoughCircles(ROI,cv2.HOUGH_GRADIENT,3.2,80,
                                    param1=lower,param2=90,minRadius=35,maxRadius=60)
        #Draw Circles
        # ensure at least some circles were found
        if circles is not None:
            
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            
            
            # loop over the (x, y) coordinates and radius of the circles
            for (y, x, r) in circles:
                if ((Xprevious - tol)<=x+xb<=(Xprevious + tol)):
                    pass
                else:
                    #Ggood+=1
                    Xprevious = x+xb
                    Yprevious = y+Ytop
                    radius=(0.5*r)              
                    start_point=round(Xprevious-radius),round(Yprevious+radius)
                    end_point= round(Xprevious+radius),round(Yprevious-radius)
                    UpperY= round(Yprevious-radius)
                    LowerY= round(Yprevious+radius)
                    UpperX= round(Xprevious+radius)
                    LowerX= round(Xprevious-radius)
                    
                    
                    ##ROI2 Definition
                    
                    ROIx=(frame1[LowerX:UpperX, UpperY:LowerY])
                    gray1 = cv2.cvtColor(ROIx, cv2.COLOR_BGR2GRAY)
                    ret1,thresh1 = cv2.threshold(gray1,120,255,cv2.THRESH_BINARY_INV)
                    ##Count pixels of suspected damage
                    cnt= (thresh1 == 0).sum()
                    if (cnt>sensitivity):
                        cv2.circle(frame, (y+Ytop, x+xb), r, (0, 0, 150), 5)
                        Bbad+=1
                        Ttotal = Bbad + Ggood
                        #client.publish(Btopic,Bbad)
                        #client.publish(Ttopic,Ttotal)
                    else:
                        cv2.circle(frame, (y+Ytop, x+xb), r, (0, 200, 0), 5)
                        Ggood+=1
                        client.publish(Gtopic,Ggood)
                        Ttotal = Bbad + Ggood
                        #client.publish(Ttopic,Ttotal)
                    cv2.imshow('Detection', thresh1)
                    print("found 1 at pos: " + str(x+xb) + " total is now: "+ str(Ggood+Bbad))
                    #cnt=len(contours)
                    cv2.imshow('ROI', ROIx)
                    print("Damage estimated: " + str(cnt) )
                    
                    time.sleep(0.3)

        ##Display the resulting frame
        #try:
        cv2.imshow('frame', frame)
        cv2.imshow('ROI', ROI)

        time.sleep(0.03)
    except:
        pass
       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("")
        print("Summary:")
        print("######################################")
        #print("TOTAL TOMATOS: "+str(Ggood))
        #print(" GOOD TOMATOS: "+str(Ggood)+"  "+"BAD TOMATOES: "+ str(Bbad))
        print("######################################")
        break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()