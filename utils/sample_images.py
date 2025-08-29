import cv2, numpy as np

def load_sample_images():
    imgs=[]
    for i in range(5):
        img = np.random.randint(100,200,(200,200,3),dtype=np.uint8)
        if i<3:
            cv2.rectangle(img,(50,50),(150,150),(80,80,80),-1)
        imgs.append(img)
    return imgs
