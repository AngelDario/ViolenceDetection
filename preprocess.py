import glob
import tqdm
import os

videos = glob.glob("DATASET/*/*/*.avi", recursive=True)

for video in tqdm.tqdm(videos):
    
    videoOpenPose = video.replace("DATASET", "DATASET_OPENPOSE")

    
    os.system(f'cd ../openpose && ./build/examples/openpose/openpose.bin --video {video} --display 0 --write_video {videoOpenPose} --disable_blending > /dev/null')
