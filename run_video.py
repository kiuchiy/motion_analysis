import sys
import cv2
import gc
import os
from sys import platform
import argparse
import time
import logging
import subprocess
import numpy as np
import pandas as pd
from logging import getLogger, StreamHandler, Formatter
from modules.motion_analysis import MotionAnalysis
from modules.track_humans import TrackHumans

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = "./" #os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


def run_video(path, video, debug):
    start_frame=0
    logger = getLogger("APP_LOG")
    stream_handler = StreamHandler()
    if debug:
        logger.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        stream_handler.setLevel(logging.INFO)
    handler_format = Formatter('%(name)s, %(levelname)s:\t%(message)s')
    stream_handler.setFormatter(handler_format)
#     logger.addHandler(stream_handler)

    # setting directories to output
    path_movie_out = os.path.join(path, 'movies_estimated')
    path_csv_estimated = os.path.join(path, 'data_estimated')
    path_png_estimated = os.path.join(path, 'png_estimated')
    csv_file = os.path.join(path_csv_estimated, video.rsplit('.')[0] + '.csv')
    os.makedirs(path_movie_out, exist_ok=True)
    os.makedirs(path_png_estimated, exist_ok=True)
    os.makedirs(path_csv_estimated, exist_ok=True)

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # open video
    cap = cv2.VideoCapture(os.path.join(path, 'movies', video))
    logger.info("OPEN: %s" % video)
    if cap.isOpened() is False:
        logger.info("ERROR: opening video stream or file")
    caps_fps = cap.get(cv2.CAP_PROP_FPS)
    ma = MotionAnalysis()
    track = TrackHumans(start_frame=start_frame)

    # processing video
    frame_no = 0
    # cmap = plt.get_cmap("tab10")
    while cap.isOpened():
        ret_val, image = cap.read()
        if not ret_val:
            break
        if frame_no == 0:
            h_pxl, w_pxl = image.shape[0], image.shape[1]

        # estimate pose
        t = time.time()
        datum = op.Datum()
#         imageToProcess = cv2.imread(image)
        datum.cvInputData = image  # imageToProcess
        opWrapper.emplaceAndPop([datum])
        time_estimation = time.time() - t
        # keypoints
        humans = datum.poseKeypoints

        # calculate cog
        t = time.time()
        bodies_cog = ma.multi_bodies_cog(humans=humans)
        bodies_cog[np.isnan(bodies_cog[:, :, :])] = 0
        humans_feature = np.concatenate((track.humans_current,
                                         bodies_cog.reshape(bodies_cog.shape[0],
                                                            bodies_cog.shape[1] * bodies_cog.shape[2])), axis=1)
        df_frame = pd.DataFrame(humans_feature.round(4))
        df_frame.to_csv(csv_file, index=False, header=None, mode='a')
        time_cog = time.time() - t
        if frame_no % int(caps_fps) == 0:
            logger.info('calculation of cog in %.4f seconds.' % time_cog)
        track.track_humans(frame_no, humans)

        # check the time to estimation
        if (frame_no % int(caps_fps)) == 0:
            logger.info("Now estimating at:" + str(int(frame_no / caps_fps)) + "[sec]")
            logger.info('inference in %.4f seconds.' % time_estimation)
            logger.debug('shape of image: ' + str(image.shape))
            logger.debug(str(humans))

        img = datum.cvOutputData
        for i in range(len(bodies_cog)):
            cv2.circle(img, (bodies_cog[i, 14, 0], bodies_cog[i, 14, 1]), 50, color=(0, 0, 0), thickness=-1)
        #     plt.vlines(bodies_cog[:, 6, 0] * w_pxl, ymin=0, ymax=h_pxl, linestyles='dashed')
        #     plt.vlines(bodies_cog[:, 7, 0] * w_pxl, ymin=0, ymax=h_pxl, linestyles='dashed')

        # for i, hum in enumerate(np.sort(track.humans_id)):
        #     df_human = track.humans_tracklet[track.humans_tracklet[:, track.clm_num] == hum]
        #     trajectories = np.array([(gdf[4 * 3 + 1] , gdf[4 * 3 + 2]) for gdf in df_human])
        #     cv2.polylines(img, [trajectories], False, (0,0,0), 3, cv2.LINE_4)
        #     trajectories = np.array([(gdf[7 * 3 + 1] , gdf[7 * 3 + 2]) for gdf in df_human])
        #     cv2.polylines(img, [trajectories], False, (0,0,0), 3, cv2.LINE_4)

        cv2.imwrite(os.path.join(path_png_estimated,
                                 video.split('.')[-2] + '{:06d}'.format(frame_no) + ".png"), img)

        # before increment, renew some args
        frame_no += 1
        gc.collect()
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    logger.info("finish estimation & start encoding")
    cmd = ["ffmpeg", "-r", str(caps_fps), "-start_number", str(start_frame),
           "-i", os.path.join(path_png_estimated, video.split('.')[-2] + "%06d.png"),
           "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           os.path.join(path_movie_out, video.split('.')[-2] + "_track.mp4")]
    subprocess.call(cmd)
    logger.debug('finished+')
