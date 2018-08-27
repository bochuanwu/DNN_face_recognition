
import os
import cv2
import sys

input_dir = './input_img/lfw'
outdir = './other_faces'
IMGSIZE = 64

if not os.path.exists(outdir):
    os.makedirs(outdir)
    

haar = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
haar1 = cv2.CascadeClassifier('./haarcascade_profileface.xml')

n=1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % n)
            img_path = path+'/'+filename
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            right_faces=cv2.flip(img,1,dst=None)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            part_faces=haar1.detectMultiScale(gray_img, 1.3, 5)
            part_faces1=haar1.detectMultiScale(right_faces, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
              
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
              
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

                n+=1
            for f_x, f_y, f_w, f_h in  part_faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
              
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
              

              
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)


                n+=1
            for f_x, f_y, f_w, f_h in  part_faces1:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
              
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
              
              
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

                n+=1
       

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
