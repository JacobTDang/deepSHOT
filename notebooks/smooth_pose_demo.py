import collections

class SimplePoseHistory:
  def __init__(self, max_frames=5):
    self.max_frames = max_frames
    self.frames = collections.deque(maxlen=max_frames)

  def add_frame(self, landmarks, confidence, frame_id):
    # store frames data inside of a dictionary
    frame_data = {
      'frame_id': frame_id,
      'landmarks': landmarks,
      'confidence': confidence
    }
    if not landmarks:
      # try interpolation immediately if there is not frame_data
      # regardless, we should just store the frame
      last_valid_frame = self.get_last_valid_frame()
      last_valid_landmark, last_valid_confidence = last_valid_frame['landmarks'], last_valid_frame['confidence']
      if not last_valid_confidence: # if confidence is null set to some default (0.3 in this case)
        last_valid_confidence = 0.3
      frame_data['landmarks'], frame_data['confidence'] = last_valid_landmark, last_valid_confidence
    # else if frame is valid
    # deque will automatically handle if self.frames length exceeds max amount
    self.frames.append(frame_data)

  def get_last_valid_frame(self):
    # this function is going to return to us the last valid frame in our queue
    for frame_data in reversed(self.frames): # O(n) time
      # if the land marks EXISTS, we return the valid frame
      if frame_data['landmarks']:
        return frame_data # return the actual frame data
      # else if there is no data
      return None

  def interpolate_missing_frame(self):
    # goal here is to get the last valid frame
    last_valid_frame = self.get_last_valid_frame()
    last_valid_landmark, last_valid_confidence = last_valid_frame['landmarks'], last_valid_frame['confidence']
    # then we fill in the gaps where the landmarks are none
    for frame_data in self.frames:
      if not frame_data['landmarks']:
        # we replace the "None" frame with a valid frame
        frame_data['landmarks'], frame_data['confidence'] = last_valid_landmark, last_valid_confidence


