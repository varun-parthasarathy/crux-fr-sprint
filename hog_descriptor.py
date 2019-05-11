import cv2
from skimage import data, exposure
from skimage.feature import hog

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        continue
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (H, h_image) = hog(image, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=True, multichannel=True)
    h_image = exposure.rescale_intensity(h_image, out_range=(0, 255))
    h_image = h_image.astype("uint8")
    cv2.imshow('Webcam Feed', frame)
    cv2.imshow('HOG', h_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()