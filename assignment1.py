import cv2


def click_event(event, x, y, flags, params):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = ""

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        text = f"{x=}, {y=}"

    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)

        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        text = f"{b=}, {g=}, {r=}"

    cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)
    cv2.imshow('image', img)


if __name__ == "__main__":
    fp = "./images/calib-checkerboard.png"
    img = cv2.imread(fp, 1)

    cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
