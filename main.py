# import cv2 as cv
# import os
# import time

# # Initialize video capture
# cap = cv.VideoCapture('rtsp://admin:admin123@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1')
# if not cap.isOpened():
#     print("Error: Unable to open camera.")
#     exit()

# # Optimize camera settings for low-light
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv.CAP_PROP_FPS, 15)

# # Create directory to save images
# output_dir = "on_hurairah"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Capture 100 images every 5 seconds
# num_images = 5
# interval = 5  # seconds

# for i in range(num_images):
#     ret, frame = cap.read()
#     if not ret:
#         print(f"Error: Unable to capture image {i+1}.")
#         continue
    
#     # Display the frame
#     cv.imshow("Live Stream", frame)
    
#     # Save image to the directory
#     image_path = os.path.join(output_dir, f"image_{i+1:03d}.jpg")
#     cv.imwrite(image_path, frame)
#     print(f"Captured and saved image {i+1} as {image_path}")
    
#     # Wait for the specified interval
#     time.sleep(interval)
    
#     # Check for key press to exit early
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting early...")
#         break

# # Release the video capture
# cap.release()
# cv.destroyAllWindows()
# print("Completed capturing 100 images.")


import cv2 as cv
import os
import time

# RTSP URL, make sure this is correct
rtsp_url = 'rtsp://admin:admin123@192.168.1.104:554/cam/realmonitor?channel=1&subtype=1'

# Initialize video capture
cap = cv.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Unable to open camera. Please check the RTSP URL and credentials.")
    exit()

# Optimize camera settings for low-light
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 15)

# Create directory to save images
output_dir = "on_hurairah"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Capture 100 images every 5 seconds
num_images = 50
interval = 5  # seconds

for i in range(num_images):
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Unable to capture image {i+1}. Please check the camera connection.")
        continue
    
    # Save image to the directory
    image_path = os.path.join(output_dir, f"image_{i+1:03d}.jpg")
    cv.imwrite(image_path, frame)
    print(f"Captured and saved image {i+1} as {image_path}")
    
    # Display the frame (optional, for monitoring)
    cv.imshow("Live Stream", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exiting early...")
        break

    # Wait for the specified interval
    time.sleep(interval)

# Release the video capture
cap.release()
cv.destroyAllWindows()
print("Completed capturing images.")
