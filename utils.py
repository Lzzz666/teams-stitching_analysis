import cv2

def load_image():
    path = "assets/test_image.png"
    image = cv2.imread(path)
    return image

def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, name):
    path = "assets/output/"
    cv2.imwrite(path + name + ".jpg", image)

