;; Auto-generated. Do not edit!


(when (boundp 'object_detection::lane_detect_try)
  (if (not (find-package "OBJECT_DETECTION"))
    (make-package "OBJECT_DETECTION"))
  (shadow 'lane_detect_try (find-package "OBJECT_DETECTION")))
(unless (find-package "OBJECT_DETECTION::LANE_DETECT_TRY")
  (make-package "OBJECT_DETECTION::LANE_DETECT_TRY"))

(in-package "ROS")
;;//! \htmlinclude lane_detect_try.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass object_detection::lane_detect_try
  :super ros::object
  :slots (_num_of_lanes _lane _num_of_points ))

(defmethod object_detection::lane_detect_try
  (:init
   (&key
    ((:num_of_lanes __num_of_lanes) 0)
    ((:lane __lane) (let (r) (dotimes (i 0) (push (instance geometry_msgs::Point :init) r)) r))
    ((:num_of_points __num_of_points) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _num_of_lanes (round __num_of_lanes))
   (setq _lane __lane)
   (setq _num_of_points __num_of_points)
   self)
  (:num_of_lanes
   (&optional __num_of_lanes)
   (if __num_of_lanes (setq _num_of_lanes __num_of_lanes)) _num_of_lanes)
  (:lane
   (&rest __lane)
   (if (keywordp (car __lane))
       (send* _lane __lane)
     (progn
       (if __lane (setq _lane (car __lane)))
       _lane)))
  (:num_of_points
   (&optional __num_of_points)
   (if __num_of_points (setq _num_of_points __num_of_points)) _num_of_points)
  (:serialization-length
   ()
   (+
    ;; int32 _num_of_lanes
    4
    ;; geometry_msgs/Point[] _lane
    (apply #'+ (send-all _lane :serialization-length)) 4
    ;; float32[] _num_of_points
    (* 4    (length _num_of_points)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _num_of_lanes
       (write-long _num_of_lanes s)
     ;; geometry_msgs/Point[] _lane
     (write-long (length _lane) s)
     (dolist (elem _lane)
       (send elem :serialize s)
       )
     ;; float32[] _num_of_points
     (write-long (length _num_of_points) s)
     (dotimes (i (length _num_of_points))
       (sys::poke (elt _num_of_points i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _num_of_lanes
     (setq _num_of_lanes (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; geometry_msgs/Point[] _lane
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _lane (let (r) (dotimes (i n) (push (instance geometry_msgs::Point :init) r)) r))
     (dolist (elem- _lane)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;; float32[] _num_of_points
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _num_of_points (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _num_of_points i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get object_detection::lane_detect_try :md5sum-) "72b1467ef9e2f8422971062049f5c7f7")
(setf (get object_detection::lane_detect_try :datatype-) "object_detection/lane_detect_try")
(setf (get object_detection::lane_detect_try :definition-)
      "int32 num_of_lanes
geometry_msgs/Point[] lane
float32[] num_of_points

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

")



(provide :object_detection/lane_detect_try "72b1467ef9e2f8422971062049f5c7f7")


