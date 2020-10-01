import cv2
#try this first with a image
#img=cv2.imread(r'C:\Users\Apoorva\Pictures\lambo.jpg')
# with webcam
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
thres=0.5 #threshold
#we have 91 names in coconames file cant write them manually so,
classNames=[]
classFile='coco.names'
with open(classFile,'rt') as f:  #rt means we are reading it
    classNames=f.read().split('\n') #strip and split each name when u see a new line and adds it to our list classNames
print(classNames)
print(len(classNames))
#download these files n u will find them in jupyter home page
configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='frozen_inference_graph.pb'
#create a model
net=cv2.dnn_DetectionModel(configPath,weightsPath)
#these below congigurations are set by default in documentation n we are going to use the same
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:  # if something is detected
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(),
                                            bbox):  # loop through each id,conf ,bbox
            cv2.rectangle(img, box, color=(0, 255, 0),
                          thickness=2)  # source img,box around it,color of box,box's thickness
            # to print the class name..*classId numbering starts from 0 so if u just type classId instead of classId-1 it will give the wrong class name
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0),
                        1)  # source,classnames frm where we get names,where u want to print the text,font,scale,color,thickness
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 1)  # to print the confidence(accuracy)
        cv2.imshow('output', img)
        cv2.waitKey(1)

