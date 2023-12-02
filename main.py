import cv2 as cv

def cap_img(img_path):
    if type(img_path) == str:
        if img_path.find('/input/test/') or img_path.find('/input/train/'):
            img = cv.imread(img_path)
            return img
        else:
            faces = cv.CascadeClassifier('check_face.xml')
            img = cv.imread(img_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            result = faces.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6)
            for (x, y, w, h) in result:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            print(result)
            new_img = img[result[0][1]:(result[0][1] + result[0][3]), result[0][0]:(result[0][0] + result[0][2])]
            return new_img
    else:
        faces = cv.CascadeClassifier('check_face.xml')
        gray = cv.cvtColor(img_path, cv.COLOR_BGR2GRAY)
        result = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        for (x, y, w, h) in result:
            cv.rectangle(img_path, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        print(result)
        new_img = img_path[result[0][1]:(result[0][1] + result[0][3]), result[0][0]:(result[0][0] + result[0][2])]
        return new_img