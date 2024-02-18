import numpy as np
import cv2
import time

# Global variables initialization
lower = 0
tol = 18
Ggood = 0
Xprevious = 0
sensitivity = 400
Bbad = 0
Ttotal = 0

def main():
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            frame_processing(frame)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print_summary()
                break

            time.sleep(0.03)

    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        cap.release()
        cv2.destroyAllWindows()

def frame_processing(frame):
    global Ggood, Xprevious, Bbad, Ttotal

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY_INV)

    # Further processing...
    # Your existing frame processing logic here...

def print_summary():
    print("")
    print("Summary:")
    print("######################################")
    print("TOTAL TOMATOES: ", Ttotal)
    print("GOOD TOMATOES: ", Ggood)
    print("BAD TOMATOES: ", Bbad)
    print("######################################")

if __name__ == "__main__":
    main()
