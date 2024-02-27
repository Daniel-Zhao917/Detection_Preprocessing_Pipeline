import cv2
import os
import easyocr
from retinaface import RetinaFace
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam

def process_frame(frame, reader):
    _, image = frame
    result = reader.readtext(image)
    id = result[0][1]
    if (id.startswith('C') and ((id[1].isdigit() or id[1] == 'o') and (id[2].isdigit() or id[2] == 'o') and (id[3].isdigit() or id[3] == 'o'))):
        # checking that the first digit for the ID is C, and following three digits are either digits or 'o' (easyOCR sometimes confuses 0 with o)
        return True, image
    return False, None

def video_to_frames(video_path, frames_dir, reader, overwrite=False, chunk_size = 1000, video_num=1):
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # get total frame count for video
    capture = cv2.VideoCapture(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = int(capture.get(cv2.CAP_PROP_FPS))
    capture.release()

    # if video has no frames
    if total < 1:
        print("Video has no frames. Check your OpenCV + ffmpeg installation")
        return None

    # per how many frames to extract face?
    extraction_rate = math.floor(vid_fps / 10)

    # chunking of frames
    frame_chunks = [[i, i + chunk_size] for i in range(0, total, chunk_size)]
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total - 1)

    for f in frame_chunks:
        capture = cv2.VideoCapture(video_path)
        capture.set(1, f[0])
        frame = f[0]

        while frame <= f[1]:
            _, image = capture.read()
            if image is None:
                break
            if frame % extraction_rate == 0: # if frame is in the increment we want
                success, processed_frame = process_frame((frame, image), reader)
                if success: # if it is a frame we want
                    resolution = processed_frame.shape
                    crop_height = int(resolution[0]*0.85) # only keep the top 85% of image (cut out interviewer's face)
                    cropped_img = processed_frame[0:crop_height, 0:resolution[1]] # cropped out bottom portion of image, aka the face we do not want
                    participant_face = RetinaFace.extract_faces(cropped_img, align = False) # get the face
                    save_path = os.path.join(frames_dir, "v{v:d}:f{f:d}.jpg".format(v = video_num, f = frame)) # name each frame as 'video number/frame.jpg'
                    if not os.path.exists(save_path) or overwrite:
                        try:
                            cv2.imwrite(save_path, participant_face[0]) # save frame to image path
                        except IndexError:
                            continue
            frame += 1
        capture.release()

    return frames_dir

if __name__ == '__main__':
    video_directory = '/Users/danielzhao/Desktop/Video_Interview_Folder'  # directory for videos
    image_path = '/Users/danielzhao/Desktop/Pipeline output'               # directory for saved images
    video_num = 1                                               # the video number currently being processed
    reader = easyocr.Reader(['en']) # initialize easyOCR
    for dirpath, dirnames, filenames in os.walk(video_directory):
        for file in filenames: # iterating through each video in the folder
            file_path = os.path.join(dirpath, file)
            video_to_frames(video_path=file_path, frames_dir=image_path, reader=reader, overwrite=False, video_num=video_num)
            video_num += 1 # add to video count so we know program is moving on to the next video
