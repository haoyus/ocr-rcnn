import cv2
import os

def video2image(vidpath):
    vidpath_split = vidpath.split('/')
    dir = vidpath_split[:-1]
    dir = '/'.join(dir)
    vidname = vidpath_split[-1].split('.')[0]

    cap = cv2.VideoCapture(vidpath)
    cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite(os.path.join(dir, '{}_{}.jpg'.format(vidname, cnt)), frame)
            cnt += 1
            break


if __name__ == '__main__':
    video2image('test_images/20220805195456.mp4')