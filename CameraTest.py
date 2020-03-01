import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('model.h5')
def get_label():
    img = cv2.resize(plt.imread('img.jpg'), (160,120))
    image = np.expand_dims(img, axis=0)
    pred = model.predict(image, batch_size=1)
    return pred[0]

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite('img.jpg', frame)
    point = get_label()
    point = (int(point[0]/160*640),int(point[1]/120*480))
    cv2.circle(frame,point, 5, (71,99,255), -1)
    # Display the resulting frame
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
