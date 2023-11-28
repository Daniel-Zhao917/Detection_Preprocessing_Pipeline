# Detection_Preprocessing_Pipeline - Daniel Zhao
(Message me on slack, or email zhaodan5@msu.edu for questions)

Current pipeline for processing video interviews of patients and interviewers and extracts the patient's faces for model training. <br />
-Change the frame_count to change the increment of frames you want to be used (e.g. frame_count = 5 to extract every 5 frames for each video). <br />

Steps of the pipeline so far: <br />
1)Iterates through each video frame by frame <br />
2)If the frame is in the increment we want, then make sure it is one where the participant's face takes up the majority of the screen with easyOCR <br />
3)Crop the image so that the interviewer's face is excluded for RetinaFace facial detection <br />
4)Use RetinaFace to detect and align the patient's face vertically <br />
4)Save the cropped face to image path<br />

*Will add autoencoder neural network architecture with ResNet-50 architecture soon, creating feature vectors of each frame
