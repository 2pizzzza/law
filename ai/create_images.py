import numpy as np
import threading
import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('Autofill_fin/best.pt')

# Define the dimensions of the rectangle
rectangle_width = 900
rectangle_height = 600
fin = False

# Define the desired video width and height
video_width = 1280
video_height = 720

cap = None

def shoot():
    global fin, cap
    # Initialize the camera
    camera = cv2.VideoCapture(0)

    # Set the camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)



    while not fin:
        # Read a frame from the camera
        ret, frame = camera.read()

        # Get the dimensions of the frame
        frame_height, frame_width = frame.shape[:2]

        # Create a mask with the same dimensions as the frame
        mask = np.zeros_like(frame, dtype=np.uint8)

        # Calculate the coordinates for the rectangle
        x = (frame_width - rectangle_width) // 2
        y = (frame_height - rectangle_height) // 2

        # Draw a white rectangle on the mask to highlight the center area
        cv2.rectangle(mask, (x, y), (x + rectangle_width, y + rectangle_height), (255, 255, 255), -1)

        # Create a mask for the area inside the rectangle
        rectangle_mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.rectangle(rectangle_mask, (x, y), (x + rectangle_width, y + rectangle_height), (255, 255, 255), -1)

        # Apply the rectangle mask to the frame
        inside_rect = cv2.bitwise_and(frame, rectangle_mask)

        # Darken the area outside the rectangle
        alpha = 0.3 
        outside_rect = cv2.addWeighted(frame, 1 - alpha, inside_rect, alpha, 0)

        # Resize the frame for display
        display_frame = cv2.resize(outside_rect, (video_width, video_height))
        cap = frame[y:y+rectangle_height, x:x+rectangle_width]

        # Display the resulting frame
        cv2.imshow('Camera', display_frame)
        if fin:
            print('True')
            break
        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()

def close():
    global fin, cap
    fin2 = True
    # Loop through the video frames
    while fin2:
        results = model(cap)
        n = 0
        previous_n = None
        Date_of_birth_found=False
        i1=True
        Date_of_expiry_found=False
        i2=True
        Document_num_found=False
        i3=True
        ID_cards_found=False
        i4=True
        Name_found=False
        i5=True
        Nationality_found=False
        i6=True
        Patronomycs_found=False   
        i7=True
        Sex_found=False
        i8=True
        Surname_found=False
        i9=True
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # count number of found classes
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                if (box.conf[0] > 0.5):
                    cls = int(box.cls[0])

                    # check the classes
                    if cls == 0:
                        Date_of_birth_found = True
                    if cls == 1:
                        Date_of_expiry_found = True
                    if cls == 2:
                        Document_num_found = True
                    if cls == 3:
                        ID_cards_found = True
                    if cls == 4:
                        Name_found = True
                    if cls == 5:
                        Nationality_found = True
                    if cls == 6:
                        Patronomycs_found = True
                    if cls == 7:
                        Sex_found = True
                    if cls == 8:
                        Surname_found = True

                # # save found classes
                # for index, item in enumerate(classes_list):
                #     if item and globals()[f"i{index+1}"]:
                #         n += 1
                #         r = box.xyxy[0].astype(int)
                #         crop = cap[r[1]:r[3], r[0]:r[2]]
                #         cv2.imwrite(str(cls) + ".jpg", crop)
                #         globals()[f"i{index+1}"] = False
                if Date_of_birth_found and i1:
                    n += 1
                    r = box.xyxy[0].astype(int)
                    crop = cap[r[1]:r[3], r[0]:r[2]]
                    cv2.imwrite('Autofill_fin/for_detecting/'+ str(cls) + ".jpg", crop)
                    i1 = False
                if Date_of_expiry_found and i2:
                    n += 1
                    r = box.xyxy[0].astype(int)
                    crop = cap[r[1]:r[3], r[0]:r[2]]
                    cv2.imwrite('Autofill_fin/for_detecting/'+ str(cls) + ".jpg", crop)
                    i2 = False
                if Document_num_found and i3:
                    n += 1
                    r = box.xyxy[0].astype(int)
                    crop = cap[r[1]:r[3], r[0]:r[2]]
                    cv2.imwrite('Autofill_fin/for_detecting/'+ str(cls) + ".jpg", crop)
                    i3 = False
                if ID_cards_found and i4:
                    n += 1
                    r = box.xyxy[0].astype(int)
                    crop = cap[r[1]:r[3], r[0]:r[2]]
                    cv2.imwrite('Autofill_fin/for_detecting/'+ str(cls) + ".jpg", crop)
                    i4 = False
                if Name_found and i5:
                    n += 1
                    r = box.xyxy[0].astype(int)
                    crop = cap[r[1]:r[3], r[0]:r[2]]
                    cv2.imwrite('Autofill_fin/for_detecting/'+ str(cls) + ".jpg", crop)
                    i5 = False
                if Nationality_found and i6:
                    n += 1
                    r = box.xyxy[0].astype(int)
                    crop = cap[r[1]:r[3], r[0]:r[2]]
                    cv2.imwrite('Autofill_fin/for_detecting/'+ str(cls) + ".jpg", crop)
                    i6 = False
                if Patronomycs_found and i7:
                    n += 1
                    r = box.xyxy[0].astype(int)
                    crop = cap[r[1]:r[3], r[0]:r[2]]
                    cv2.imwrite('Autofill_fin/for_detecting/'+ str(cls) + ".jpg", crop)
                    i7 = False
                if Sex_found and i8:
                    n += 1
                    r = box.xyxy[0].astype(int)
                    crop = cap[r[1]:r[3], r[0]:r[2]]
                    cv2.imwrite('Autofill_fin/for_detecting/'+ str(cls) + ".jpg", crop)
                    i8 = False
                if Surname_found and i9:
                    n += 1
                    r = box.xyxy[0].astype(int)
                    crop = cap[r[1]:r[3], r[0]:r[2]]
                    cv2.imwrite('Autofill_fin/for_detecting/'+ str(cls) + ".jpg", crop)
                    i9 = False
            # print n when its value changes
            if n != previous_n:
                print("n:", n)
                previous_n = n
                print(n)
        # break if all classes found
        if n == 9:
            print('Document successfully scanned!')
            fin2 = False

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("b"):
            break

    # while True:
    #     cv2.imshow('Captured Image', captured_image)

    #     # Check for the 'b' key to exit the loop
    #     if cv2.waitKey(1) & 0xFF == ord('b'):
    #         break

    print('stop')
    fin = True

if __name__ == '__main__':
    p1 = threading.Thread(target=shoot)
    p2 = threading.Thread(target=close)

    p1.start()
    time.sleep(10)
    p2.start()

    p2.join()
    p1.join()
