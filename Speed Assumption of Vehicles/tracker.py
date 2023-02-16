import cv2
import numpy as np
from object_detection import ObjectDetection
import math

od = ObjectDetection()

cap = cv2.VideoCapture('input/video1.mp4')
ret, frame = cap.read()
#rows, cols, _ = img.shape

count = 0
center_points_prev_frame = []

tracking_obj = {}
track_id = 0

area_1 = [(234,692),(1263,691),(1108,440),(431,407)]
area_2 = [(603,172),(979,167),(909,74),(682,67)]

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    #img = img[356: rows, 552: cols]
    center_points_cur_frame = []

    #Detection
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w)/2)
        cy = int((y + y + h)/2)

        center_points_cur_frame.append((cx,cy))

        result = cv2.pointPolygonTest(np.array(area_1,np.int32),(int(cx),int(cy)), False)
        if result>=0:
            print('Frame no.: ',count,'Box: ',x,y,w,h)
            #cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)

    #print('Current Frame: ')
    #print(center_points_cur_frame)
#
    #print('Previos Frame: ')
    #print(center_points_prev_frame)

    if count <= 2:
        for pt in center_points_cur_frame:
            #cv2.circle(frame, pt, 3, (0, 0, 255), -1)
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])

                if distance < 20:
                    tracking_obj[track_id] = pt
                    track_id += 1

    else:
        tracking_objects_copy = tracking_obj.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                # Update IDs position
                if distance < 40:
                    tracking_obj[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue
            # Remove IDs lost
            if not object_exists:
                tracking_obj.pop(object_id)
        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_obj[track_id] = pt
            track_id += 1
    
    
    #for object_id, pt in tracking_obj.items():
    #    cv2.circle(frame, pt, 3, (0,0,255), -1)
    #    cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0 ,1, (0, 0, 255), 1)
    
    #print(tracking_obj)
    for i, area in enumerate([area_1,area_2]):
        if i == 0:
            cv2.polylines(frame, [np.array(area, np.int32)], True, (15,220,10), 6)   


    cv2.imshow('frame',frame)
    
    center_points_prev_frame = center_points_cur_frame.copy()
    
    key = cv2.waitKey(33)
    if key == 27:
        break

cv2.destroyAllWindows()