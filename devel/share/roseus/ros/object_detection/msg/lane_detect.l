;; Auto-generated. Do not edit!


(when (boundp 'object_detection::lane_detect)
  (if (not (find-package "OBJECT_DETECTION"))
    (make-package "OBJECT_DETECTION"))
  (shadow 'lane_detect (find-package "OBJECT_DETECTION")))
(unless (find-package "OBJECT_DETECTION::LANE_DETECT")
  (make-package "OBJECT_DETECTION::LANE_DETECT"))

(in-package "ROS")
;;//! \htmlinclude lane_detect.msg.html


(defclass object_detection::lane_detect
  :super ros::object
  :slots (_lanes ))

(defmethod object_detection::lane_detect
  (:init
   (&key
    ((:lanes __lanes) (let (r) (dotimes (i 0) (push (instance object_detection::lane_detect_try :init) r)) r))
    )
   (send-super :init)
   (setq _lanes __lanes)
   self)
  (:lanes
   (&rest __lanes)
   (if (keywordp (car __lanes))
       (send* _lanes __lanes)
     (progn
       (if __lanes (setq _lanes (car __lanes)))
       _lanes)))
  (:serialization-length
   ()
   (+
    ;; object_detection/lane_detect_try[] _lanes
    (apply #'+ (send-all _lanes :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; object_detection/lane_detect_try[] _lanes
     (write-long (length _lanes) s)
     (dolist (elem _lanes)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; object_detection/lane_detect_try[] _lanes
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _lanes (let (r) (dotimes (i n) (push (instance object_detection::lane_detect_try :init) r)) r))
     (dolist (elem- _lanes)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get object_detection::lane_detect :md5sum-) "4a43b030de32eaaba0005ebbcbcc1fca")
(setf (get object_detection::lane_detect :datatype-) "object_detection/lane_detect")
(setf (get object_detection::lane_detect :definition-)
      "lane_detect_try[] lanes

================================================================================
MSG: object_detection/lane_detect_try
int32 num_of_lanes
geometry_msgs/Point[] lane
float32[] num_of_points

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

")



(provide :object_detection/lane_detect "4a43b030de32eaaba0005ebbcbcc1fca")


