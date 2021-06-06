
(cl:in-package :asdf)

(defsystem "object_detection-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
)
  :components ((:file "_package")
    (:file "Bounds" :depends-on ("_package_Bounds"))
    (:file "_package_Bounds" :depends-on ("_package"))
    (:file "BoundsBoxes" :depends-on ("_package_BoundsBoxes"))
    (:file "_package_BoundsBoxes" :depends-on ("_package"))
    (:file "coordinate_pairs_lane" :depends-on ("_package_coordinate_pairs_lane"))
    (:file "_package_coordinate_pairs_lane" :depends-on ("_package"))
    (:file "lane_detect" :depends-on ("_package_lane_detect"))
    (:file "_package_lane_detect" :depends-on ("_package"))
    (:file "lane_detect_try" :depends-on ("_package_lane_detect_try"))
    (:file "_package_lane_detect_try" :depends-on ("_package"))
  ))