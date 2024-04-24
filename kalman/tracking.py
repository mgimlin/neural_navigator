import time

from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

from kalman import predict_next_position, update_position



# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')
model = YOLO('../../yolo/gator_results/weights/best.pt')

# model = YOLO('yolov8n-seg.pt')  # Load an official Segment model

# Open the video file
video_path = "./test3.mp4"
cap = cv2.VideoCapture(0)

# Store the track history
track_history = defaultdict(lambda: [])
vanish_count = defaultdict(lambda: 0)
ghost = defaultdict(lambda: True)



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    curr_tracks = defaultdict(lambda: ())

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # if(not results[0].boxes.id):
        #     print("HHHkk")
        #     continue

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box

            track = track_history[track_id] # default dict creates empty list at this key


            curr_track = curr_tracks[track_id]
            

            # if (not track) or (len(track) == 1): #empty or only has 1 element
            #     predicted = predict_next_position([(float(x), float(y))])
            # else:
            #     print(track)
            #     predicted = predict_next_position(track)
            # print("ID:", track_id, "coordinates", float(x), float(y), "predicted", predicted)

            curr_tracks[track_id] = (float(x), float(y))  # x, y center point

            # track.append(curr_tracks[track_id])  # x, y center point

            # # Draw the tracking lines
            # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            


        for track_id, coordinates_history in track_history.items():


            track = track_history[track_id]
            curr_track = curr_tracks[track_id]

            if not track: # If track is empty but is showing up in track history, it was just detected for the first time
                predicted = curr_track


            elif len(track) < 18: # not enough to interpolate
                if not curr_track: # showed up once and vanished after a few frames. will assume it was a ghost detection
                    # track_history.pop(track_id) # delete ghost from record
                    print("HERE2")
                    track = []
                    vanish_count[track_id] = -1

                    continue
                else: # detected. still using measured values only
                    predicted = curr_track

            elif len(track) >= 18: # enough to interpolate
                if not curr_track:
                    print("GONE")
                    vanish_count[track_id] += 1
                    if vanish_count[track_id] > 5:
                        continue
                    predicted = predict_next_position(track)
                else:
                    print("NO VANISH")
                    predicted = predict_next_position(track)
                    predicted = update_position(curr_track, predicted)


            # elif len(track) == 1: # 1 previous track
            #     if not curr_track: # showed up once and vanished after a frame. will assume it was a ghost detection
            #         # track_history.pop(track_id) # delete ghost from record
            #         print("HERE2")

            #         continue
            #     else: # detected. still using measured values only
            #         predicted = curr_tracks[track_id]
                    
            # else: # with 2 or more points, we can now use kalman to predict

            #     if not curr_track: # No detction for this object in the current frame. Either it has left the frame or it failed to pick up
                    
            #         predicted = predict_next_position(track)
            #         # check if predicted coordinates are within frame. If it has left, use track_history.pop(track_id) and continue


            #     else:
            #         predicted = predict_next_position(track)
            #         predicted = update_position(curr_track, predicted)


            # if (not track) or (len(track) == 1): #empty or only has 1 element
            # if len(track) < 18: #not enough positions to interpolate
            #     predicted = curr_track
            #     print("HERE1", predicted)
            # else:
            #     # print(track)
            #     predicted = predict_next_position(track)
            #     predicted = update_position(curr_track, predicted)
            
            print("ID:", track_id, "coordinates", curr_track, "predicted", predicted)

            track.append(predicted)  # x, y center point


            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

             

            


        # # Plot the tracks
        # for box, track_id in zip(boxes, track_ids):
        #     x, y, w, h = box
        #     track = track_history[track_id]
        #     if (not track) or (len(track) == 1): #empty or only has 1 element
        #         predicted = predict_next_position([(float(x), float(y))])
        #     else:
        #         predicted = predict_next_position(track)
                 

        #     print("ID:", track_id, "coordinates", float(x), float(y), "predicted", predicted)

        #     track.append((float(x), float(y)))  # x, y center point

        #     # track.append((float(x), float(y)))  # x, y center point
        #     # print("ID:", track_id, "coordinates", float(x), float(y), "predicted", predict_next_position(track))

            
        #     if len(track) > 30:  # retain 90 tracks for 90 frames
        #         track.pop(0)

        #     # Draw the tracking lines
        #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    # time.sleep(10)

# Release the video capture object and close the display windowcc 
cap.release()
cv2.destroyAllWindows()
