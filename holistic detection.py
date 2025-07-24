import cv2
import mediapipe as mp

mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic()

mpDraw = mp.solutions.drawing_utils
drawing_specs = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = holistic.process(imgRGB)

    if results.face_landmarks:
        mpDraw.draw_landmarks(img, results.face_landmarks, mpHolistic.FACEMESH_CONTOURS, drawing_specs, drawing_specs)

    if results.left_hand_landmarks:
        mpDraw.draw_landmarks(img, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS, drawing_specs, drawing_specs)

    if results.right_hand_landmarks:
        mpDraw.draw_landmarks(img, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS, drawing_specs, drawing_specs)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS, drawing_specs, drawing_specs)


    cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
