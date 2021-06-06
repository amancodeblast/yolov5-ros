; Auto-generated. Do not edit!


(cl:in-package object_detection-msg)


;//! \htmlinclude lane_detect_try.msg.html

(cl:defclass <lane_detect_try> (roslisp-msg-protocol:ros-message)
  ((num_of_lanes
    :reader num_of_lanes
    :initarg :num_of_lanes
    :type cl:integer
    :initform 0)
   (lane
    :reader lane
    :initarg :lane
    :type (cl:vector geometry_msgs-msg:Point)
   :initform (cl:make-array 0 :element-type 'geometry_msgs-msg:Point :initial-element (cl:make-instance 'geometry_msgs-msg:Point)))
   (num_of_points
    :reader num_of_points
    :initarg :num_of_points
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass lane_detect_try (<lane_detect_try>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <lane_detect_try>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'lane_detect_try)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name object_detection-msg:<lane_detect_try> is deprecated: use object_detection-msg:lane_detect_try instead.")))

(cl:ensure-generic-function 'num_of_lanes-val :lambda-list '(m))
(cl:defmethod num_of_lanes-val ((m <lane_detect_try>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection-msg:num_of_lanes-val is deprecated.  Use object_detection-msg:num_of_lanes instead.")
  (num_of_lanes m))

(cl:ensure-generic-function 'lane-val :lambda-list '(m))
(cl:defmethod lane-val ((m <lane_detect_try>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection-msg:lane-val is deprecated.  Use object_detection-msg:lane instead.")
  (lane m))

(cl:ensure-generic-function 'num_of_points-val :lambda-list '(m))
(cl:defmethod num_of_points-val ((m <lane_detect_try>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection-msg:num_of_points-val is deprecated.  Use object_detection-msg:num_of_points instead.")
  (num_of_points m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <lane_detect_try>) ostream)
  "Serializes a message object of type '<lane_detect_try>"
  (cl:let* ((signed (cl:slot-value msg 'num_of_lanes)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'lane))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'lane))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'num_of_points))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'num_of_points))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <lane_detect_try>) istream)
  "Deserializes a message object of type '<lane_detect_try>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'num_of_lanes) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'lane) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'lane)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'geometry_msgs-msg:Point))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'num_of_points) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'num_of_points)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<lane_detect_try>)))
  "Returns string type for a message object of type '<lane_detect_try>"
  "object_detection/lane_detect_try")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'lane_detect_try)))
  "Returns string type for a message object of type 'lane_detect_try"
  "object_detection/lane_detect_try")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<lane_detect_try>)))
  "Returns md5sum for a message object of type '<lane_detect_try>"
  "72b1467ef9e2f8422971062049f5c7f7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'lane_detect_try)))
  "Returns md5sum for a message object of type 'lane_detect_try"
  "72b1467ef9e2f8422971062049f5c7f7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<lane_detect_try>)))
  "Returns full string definition for message of type '<lane_detect_try>"
  (cl:format cl:nil "int32 num_of_lanes~%geometry_msgs/Point[] lane~%float32[] num_of_points~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'lane_detect_try)))
  "Returns full string definition for message of type 'lane_detect_try"
  (cl:format cl:nil "int32 num_of_lanes~%geometry_msgs/Point[] lane~%float32[] num_of_points~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <lane_detect_try>))
  (cl:+ 0
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'lane) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'num_of_points) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <lane_detect_try>))
  "Converts a ROS message object to a list"
  (cl:list 'lane_detect_try
    (cl:cons ':num_of_lanes (num_of_lanes msg))
    (cl:cons ':lane (lane msg))
    (cl:cons ':num_of_points (num_of_points msg))
))
