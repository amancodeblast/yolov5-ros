// Auto-generated. Do not edit!

// (in-package object_detection.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class lane_detect_try {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.num_of_lanes = null;
      this.lane = null;
      this.num_of_points = null;
    }
    else {
      if (initObj.hasOwnProperty('num_of_lanes')) {
        this.num_of_lanes = initObj.num_of_lanes
      }
      else {
        this.num_of_lanes = 0;
      }
      if (initObj.hasOwnProperty('lane')) {
        this.lane = initObj.lane
      }
      else {
        this.lane = [];
      }
      if (initObj.hasOwnProperty('num_of_points')) {
        this.num_of_points = initObj.num_of_points
      }
      else {
        this.num_of_points = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type lane_detect_try
    // Serialize message field [num_of_lanes]
    bufferOffset = _serializer.int32(obj.num_of_lanes, buffer, bufferOffset);
    // Serialize message field [lane]
    // Serialize the length for message field [lane]
    bufferOffset = _serializer.uint32(obj.lane.length, buffer, bufferOffset);
    obj.lane.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Point.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [num_of_points]
    bufferOffset = _arraySerializer.float32(obj.num_of_points, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type lane_detect_try
    let len;
    let data = new lane_detect_try(null);
    // Deserialize message field [num_of_lanes]
    data.num_of_lanes = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [lane]
    // Deserialize array length for message field [lane]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.lane = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.lane[i] = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [num_of_points]
    data.num_of_points = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 24 * object.lane.length;
    length += 4 * object.num_of_points.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'object_detection/lane_detect_try';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '72b1467ef9e2f8422971062049f5c7f7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int32 num_of_lanes
    geometry_msgs/Point[] lane
    float32[] num_of_points
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new lane_detect_try(null);
    if (msg.num_of_lanes !== undefined) {
      resolved.num_of_lanes = msg.num_of_lanes;
    }
    else {
      resolved.num_of_lanes = 0
    }

    if (msg.lane !== undefined) {
      resolved.lane = new Array(msg.lane.length);
      for (let i = 0; i < resolved.lane.length; ++i) {
        resolved.lane[i] = geometry_msgs.msg.Point.Resolve(msg.lane[i]);
      }
    }
    else {
      resolved.lane = []
    }

    if (msg.num_of_points !== undefined) {
      resolved.num_of_points = msg.num_of_points;
    }
    else {
      resolved.num_of_points = []
    }

    return resolved;
    }
};

module.exports = lane_detect_try;
