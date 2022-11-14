import cv2
import pickle

facedetect=cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
eyedetect=cv2.CascadeClassifier('models/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={}
with open("labels.pickle", "rb") as f:
  og_labels=pickle.load(f)
  labels={v:k for k,v in og_labels.items()}
video=cv2.VideoCapture(0)
# count=0
while True:
  ret,frame= video.read()
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces=facedetect.detectMultiScale(gray,1.3,5)

  for x,y,w,h in faces:
    # count= count+1
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    # name="susee"+str(count)+".png"
    #for gray
    roi_gray=gray[y:y+h,x:x+w]
    #for color
    roi_color=frame[y:y+h,x:x+w]

    #prediction, deep learning technique
    id_,conf = recognizer.predict(roi_gray)
    if conf>=45 and conf <=85:
      # print(id_)
      # print(labels[id_])
      font = cv2.FONT_HERSHEY_SIMPLEX
      name=labels[id_]
      color=(255,255,255)
      stroke=2
      cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
    img_item="train_image.png"
    cv2.imwrite(img_item,roi_color)

    #Find eyes------------------------------------------------------------------
    color = (255, 0, 0)
    stroke = 2
    eyes = eyedetect.detectMultiScale(roi_color)
    for ex,ey,ew,eh in eyes:
      cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    ## Insert liveness detector through eye blinking code here
  


  #Display the resulting frame
  cv2.imshow("frame",frame)
  k=cv2.waitKey(1)
  if k==ord("q"):
    break
video.release()
cv2.destroyAllWindows()