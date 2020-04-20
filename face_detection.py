import cv2

#Create Haar cascade
face_cascade = cv2.CascadeClassifier('/home/neha/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('/home/neha/Downloads/nurr.jpg')

#Convert RGB to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#scalefactor = 1.1
#minneighbours = 4
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#Draw rectangle around faces
for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()
