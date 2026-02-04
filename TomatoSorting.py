import cv2
import numpy as np
import time


# =====================
# Configuration
# =====================
TOLERANCE = 18
SENSITIVITY = 400
FRAME_SCALE = 0.3

# =====================
# Counters
# =====================
good_count = 0
bad_count = 0
previous_x = None


# =====================
# Utility Functions
# =====================
def setup_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def resize_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY_INV)
    return gray, thresh


def get_roi(thresh, frame_shape):
    xb = 5
    y_top = int(frame_shape[1] * 0.5)
    y_bottom = int(frame_shape[1] * 0.7)
    width = frame_shape[0]
    roi = thresh[xb:width - xb, y_top:y_bottom]
    return roi, xb, y_top, y_bottom


def auto_canny_lower(gray):
    v = np.median(gray)
    sigma = 0.33
    return int(max(0, (1.0 - sigma) * v))


def morphological_ops(thresh):
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=3)
    erosion = cv2.erode(dilation, kernel, iterations=2)
    return erosion


def detect_circles(roi, canny_lower):
    return cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=3.2,
        minDist=80,
        param1=canny_lower,
        param2=90,
        minRadius=35,
        maxRadius=60
    )


def estimate_damage(frame, center_x, center_y, radius):
    r = int(0.5 * radius)
    x1, x2 = center_x - r, center_x + r
    y1, y2 = center_y - r, center_y + r

    roi = frame[x1:x2, y1:y2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    damage_pixels = (thresh == 0).sum()
    return damage_pixels, roi, thresh


# =====================
# Main Processing Loop
# =====================
def main():
    global good_count, bad_count, previous_x

    cap = setup_camera()

    ret, frame = cap.read()
    if not ret:
        print("Failed to initialize camera")
        return

    previous_x = frame.shape[1]

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_frame(frame, FRAME_SCALE)
            frame_copy = frame.copy()

            gray, thresh = preprocess_frame(frame)
            roi, xb, y_top, y_bottom = get_roi(thresh, frame.shape)

            # Draw ROI box
            cv2.rectangle(
                frame,
                (y_top, xb),
                (y_bottom, frame.shape[0] - xb),
                (0, 128, 255),
                2
            )

            canny_lower = auto_canny_lower(gray)
            processed = morphological_ops(thresh)

            circles = detect_circles(roi, canny_lower)

            if circles is not None:
                circles = np.round(circles[0]).astype(int)

                for (y, x, r) in circles:
                    current_x = x + xb

                    if abs(previous_x - current_x) <= TOLERANCE:
                        continue

                    previous_x = current_x
                    center_y = y + y_top

                    damage, tomato_roi, damage_mask = estimate_damage(
                        frame_copy,
                        current_x,
                        center_y,
                        r
                    )

                    if damage > SENSITIVITY:
                        color = (0, 0, 150)
                        bad_count += 1
                    else:
                        color = (0, 200, 0)
                        good_count += 1

                    cv2.circle(frame, (center_y, current_x), r, color, 5)

                    print(
                        f"Detected tomato | Damage: {damage} | "
                        f"Total: {good_count + bad_count}"
                    )

                    cv2.imshow("Tomato ROI", tomato_roi)
                    cv2.imshow("Damage Mask", damage_mask)

                    time.sleep(0.3)

            cv2.imshow("Frame", frame)
            cv2.imshow("ROI", roi)

            time.sleep(0.03)

        except Exception:
            pass

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print_summary()
    cap.release()
    cv2.destroyAllWindows()


def print_summary():
    print("\nSummary:")
    print("############################")
    print(f"Total Tomatoes: {good_count + bad_count}")
    print(f"Good Tomatoes : {good_count}")
    print(f"Bad Tomatoes  : {bad_count}")
    print("############################")


if __name__ == "__main__":
    main()
