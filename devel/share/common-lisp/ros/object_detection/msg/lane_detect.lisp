; Auto-generated. Do not edit!


(cl:in-package object_detection-msg)


;//! \htmlinclude lane_detect.msg.html

(cl:defclass <lane_detect> (roslisp-msg-protocol:ros-message)
  ((lanes
    :reader lanes
    :initarg :lanes
    :type (cl:vector object_detection-msg:lane_detect_try)
   :initform (cl:make-array 0 :element-type 'object_detection-msg:lane_detect_try :initial-element (cl:make-instance 'object_detection-msg:lane_detect_try))))
)

(cl:defclass lane_detect (<lane_detect>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <lane_detect>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'lane_detect)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name object_detection-msg:<lane_detect> is deprecated: use object_detection-msg:lane_detect instead.")))

(cl:ensure-generic-function 'lanes-val :lambda-list '(m))
(cl:defmethod lanes-val ((m <lane_detect>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection-msg:lanes-val is deprecated.  Use object_detection-msg:lanes instead.")
  (lanes m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <lane_detect>) ostream)
  "Serializes a message object of type '<lane_detect>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'lanes))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'lanes))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <lane_detect>) istream)
  "Deserializes a message object of type '<lane_detect>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'lanes) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'lanes)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'object_detection-msg:lane_detect_try))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<lane_detect>)))
  "Returns string type for a message object of type '<lane_detect>"
  "object_detection/lane_detect")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'lane_detect)))
  "Returns string type for a message object of type 'lane_detect"
  "object_detection/lane_detect")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<lane_detect>)))
  "Returns md5sum for a message object of type '<lane_detect>"
  "4a43b030de32eaaba0005ebbcbcc1fca")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'lane_detect)))
  "Returns md5sum for a message object of type 'lane_detect"
  "4a43b030de32eaaba0005ebbcbcc1fca")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<lane_detect>)))
  "Returns full string definition for message of type '<lane_detect>"
  (cl:format cl:nil "lane_detect_try[] lanes~%~%================================================================================~%MSG: object_detection/lane_detect_try~%int32 num_of_lanes~%geometry_msgs/Point[] lane~%float32[] num_of_points~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'lane_detect)))
  "Returns full string definition for message of type 'lane_detect"
  (cl:format cl:nil "lane_detect_try[] lanes~%~%================================================================================~%MSG: object_detection/lane_detect_try~%int32 num_of_lanes~%geometry_msgs/Point[] lane~%float32[] num_of_points~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <lane_detect>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'lanes) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <lane_detect>))
  "Converts a ROS message object to a list"
  (cl:list 'lane_detect
    (cl:cons ':lanes (lanes msg))
))
