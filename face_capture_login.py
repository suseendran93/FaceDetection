import cv2
import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

image_dir=os.path.join(BASE_DIR,"images_login")
person_name=input("Enter your name: ")
os.mkdir(image_dir+"/"+person_name)
real_path=os.path.join(image_dir,person_name)
facedetect=cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
video=cv2.VideoCapture(0)
count=0

while True:
  ret,frame= video.read()
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces=facedetect.detectMultiScale(gray,1.3,5)

  for x,y,w,h in faces:
    count= count+1
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    name=person_name+str(count)+".png"
    #for gray
    roi_gray=gray[y:y+h,x:x+w]
    #for color
    roi_color=frame[y:y+h,x:x+w]
    cv2.imwrite(real_path+"/"+name,roi_color)


  #Display the resulting frame
  cv2.imshow("frame",frame)
  k=cv2.waitKey(20)
  if count>200:
    break
video.release()
cv2.destroyAllWindows()