import glob
import tqdm
import os

videos = glob.glob("DATASET/*/*/*.avi", recursive=True)
begin_path = "/home/angeldiaz/preprocess/"


#os.system(f'cd ../openpose && ./build/examples/openpose/openpose.bin --video /home/angeldiaz/preprocess/DATASET/train/Fight/_2RYnSFPD_U_0.avi --display 0 --write_video /home/angeldiaz/preprocess/_2RYnSFPD_U_0.avi --disable_blending > /dev/null --net_resolution 160x80')

for video in tqdm.tqdm(videos):
    
    videoOpenPose = video.replace("DATASET", "DATASET_OPENPOSE")
    save_path = os.path.dirname(videoOpenPose)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tqdm.tqdm.write("Processing: " + video)
    os.system(f'cd ../openpose && ./build/examples/openpose/openpose.bin --video {begin_path + video} --display 0 --write_video {begin_path + videoOpenPose} --disable_blending > /dev/null --net_resolution 160x80')
