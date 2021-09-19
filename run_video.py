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
from modules.humans_to_array import calc_torso_length
from importlib import reload as reload
from modules.draw_cv import dotline, polydotline
import matplotlib.pyplot as plt


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


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def run_video(video, path='', skip_cog=False, skip_track=False, plt_graph=False, graph_num=0,
              cog_color='black', start_frame=0, ratio_band=0.25, debug=False):
    try:
        logging.shutdown()
        reload(logging)
    except Exception as e:
        raise e
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
    logger.addHandler(stream_handler)

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
    ma = MotionAnalysis(fps=caps_fps, start_frame=start_frame)
    # CSV FILE SETTING
    segments = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "background",
                "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
                "head_cog", "torso_cog", "r_thigh_cog", "l_thigh_cog", "r_leg_cog", "l_leg_cog", "r_foot_cog", "l_foot_cog",
                "r_arm_cog", "l_arm_cog", "r_forearm_cog", "l_forearm_cog", "r_hand_cog", "l_hand_cog"]
    seg_columns = ['frame', 'human_id']
    [seg_columns.extend([x + '_x', x + '_y', x + '_score']) for x in segments]
    df_template = pd.DataFrame(columns=seg_columns)
    df_template.to_csv(csv_file, index=False)

    # processing video
    frame_no = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        if not ret_val:
            break
        if frame_no == 0:
            h_pxl, w_pxl = image.shape[0], image.shape[1]
        if frame_no < start_frame:
            frame_no += 1
            continue

        # estimate pose
        t = time.time()
        datum = op.Datum()
        datum.cvInputData = image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        time_estimation = time.time() - t
        # keypoints
        humans = datum.poseKeypoints

        # calculate and save data
        t = time.time()
        ma.track_humans(frame_no, humans)
        # calculate cog
        bodies_cog = ma.bodies_cog
        df_frame = pd.DataFrame(ma.humans_current.round(1))
        df_frame.to_csv(csv_file, index=False, header=None, mode='a')
        time_cog = time.time() - t

        if frame_no % int(caps_fps) == 0:
            # for resetting number of graph.
            hum_count = len(ma.humans_id)

        # PLOT Pictures for movie
        img = datum.cvOutputData

        # plot cog & foot lines
        if len(humans.shape) != 0:
            cog_size = (calc_torso_length(humans) / 8).astype(int)

        for i, hum in enumerate(np.sort(ma.humans_id)):
            hum_track = ma.humans_tracklet[ma.humans_tracklet[:, 1] == hum]
            hum_track = hum_track.astype(int)
            # plot cog & foot lines
            if not skip_cog:
                cv2.circle(img, (bodies_cog[i, 14, 0], bodies_cog[i, 14, 1]), cog_size[i, i], color=(0, 0, 0), thickness=-1)
                cv2.ellipse(img, center=(bodies_cog[i, 14, 0], bodies_cog[i, 14, 1]),
                            axes=(cog_size[i, i], cog_size[i, i]), angle=0, startAngle=0, endAngle=90,
                            color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_8)
                cv2.ellipse(img, center=(bodies_cog[i, 14, 0], bodies_cog[i, 14, 1]),
                            axes=(cog_size[i, i], cog_size[i, i]), angle=0, startAngle=180, endAngle=270,
                            color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_8)
                # plot foot line
                if bodies_cog[i, 6, 0]:
                    dotline(img, (bodies_cog[i, 6, 0], 0), (bodies_cog[i, 6, 0], h_pxl), color=(10, 10, 10), thickness=2)
                if bodies_cog[i, 7, 0]:
                    dotline(img, (bodies_cog[i, 7, 0], 0), (bodies_cog[i, 7, 0], h_pxl), color=(10, 10, 10), thickness=2)
                # trajectories of COG
                trajectories = np.array([(pos[39 * 3 + 2], pos[39 * 3 + 3]) for pos in hum_track if pos[39 * 3 + 2] > 1])
                cv2.polylines(img, [trajectories], False, (0, 0, 0), 3, cv2.LINE_4)

            # plot hands trajectories
            if not skip_track:
                trajectories = np.array([(pos[4 * 3 + 2], pos[4 * 3 + 3]) for pos in hum_track if pos[4 * 3 + 2] > 1])
                cv2.polylines(img, [trajectories], False, (200, 200, int(hum%3)*30), 3, cv2.LINE_4)
                trajectories = np.array([(pos[7 * 3 + 2], pos[7 * 3 + 3]) for pos in hum_track if pos[7 * 3 + 2] > 1])
                cv2.polylines(img, [trajectories], False, (int(hum%3)*30, 200, int(hum%3)*30), 3, cv2.LINE_4)

        # plt graph of cog rate
        if not plt_graph:
            # save figure
            cv2.imwrite(os.path.join(path_png_estimated,
                                     video.split('.')[-2] + '{:06d}'.format(frame_no) + ".png"), img)
        else:
            if graph_num:
                graph_row = graph_num if graph_num < 6 else 6
                graph_col = graph_num + (1 if graph_num < 6 else int(graph_num/6)+1)
            else:
                graph_row = 6  # if hum_count > 6 else 4
                graph_col = graph_row + 2 + int((hum_count-graph_row if hum_count-graph_row > 0 else 0)/graph_row)
            fig = plt.figure(figsize=(16, 8))
            grid_size = (graph_row, graph_col)
            ax_img = plt.subplot2grid(grid_size, (0, 0), rowspan=graph_row, colspan=graph_row)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax_img.imshow(img)
            if len(ma.humans_id):
                hum_idx = ~np.isnan(ma.humans_current[:, 19 * 3 + 2])
                ax_img.set_xticks(ma.humans_current[hum_idx, 19 * 3 + 2])
                ax_img.set_xticklabels(list((ma.humans_id[hum_idx]).astype(str)))

            tmin, tmax = frame_no - caps_fps, frame_no
            for i, hum in enumerate(np.sort(ma.humans_id)):
                if i == (graph_row * (graph_col - graph_row)):
                    break  # count of humans is over the capacity
                hum_track = ma.humans_tracklet[ma.humans_tracklet[:, 1] == hum]
                hum_track = hum_track.astype(int)
                ax_graph = plt.subplot2grid(grid_size, (i - graph_row * int(i / graph_row), graph_row + int(i / graph_row)))
                hum_track = hum_track[~np.isnan(hum_track[:, 19 * 3 + 2])]
                hum_track = hum_track[~np.isnan(hum_track[:, 22 * 3 + 2])]
                if hum_track.shape[0] > 0:
                    foot = (hum_track[:, 39 * 3 + 2] - hum_track[:, 19 * 3 + 2]) / (hum_track[:, 22 * 3 + 2] - hum_track[:, 19 * 3 + 2])
                    line1 = ax_graph.plot(hum_track[:, 0], foot)
                    p0 = ax_graph.hlines([0.5], tmin, tmax, "blue", linestyles='dashed')  # hlines
                    p1 = ax_graph.hlines([0], tmin, tmax, "blue", linestyles='dashed')  # hlines
                    p2 = ax_graph.hlines([1], tmin, tmax, "blue", linestyles='dashed')  # hlines
                    ax_graph.legend([str(hum)])
                    ax_graph.set_xticks([int(tmin/caps_fps*2)*caps_fps/2, int(tmin/caps_fps*2)*caps_fps/2 + caps_fps/2, int(tmax/caps_fps*2)*caps_fps/2])
                    ax_graph.set_xticklabels(list(map(str, [int(tmin/caps_fps*2)/2, int(tmin/caps_fps*2)/2+0.5, int(tmax/caps_fps*2)/2])))
                    ax_graph.set_ylim([-ratio_band, 1+ratio_band])
                    ax_graph.set_xlim([tmin, tmax])
            plt.savefig(os.path.join(path_png_estimated,
                                     video.split('.')[-2] + '{:06d}'.format(frame_no) + ".png"))
            plt.close()
            plt.clf()

        # logging to check progress and speed
        if frame_no % int(caps_fps) == 0:
            logger.info('calculation of cog in %.4f seconds.' % time_cog)
            logger.info("Now estimating at:" + str(int(frame_no / caps_fps)) + "[sec]")
            logger.info('inference in %.4f seconds.' % time_estimation)
            logger.debug('shape of image: ' + str(image.shape))
            logger.debug(str(humans))
            logger.info('shape of humans: ' + str(humans.shape))

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
           os.path.join(path_movie_out, video.split('.')[-2] + ".mp4")]
    subprocess.call(cmd)
    logger.debug('finished+')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--graph_num', type=int, default=0)
    parser.add_argument('--ratio_band', type=float, default=0.25)
    parser.add_argument('--skip_track', type=bool, default=False)
    parser.add_argument('--plt_graph', type=bool, default=False)
    parser.add_argument('--skip_cog', type=bool, default=False)
    parser.add_argument('--cog_color', type=str, default='black')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    run_video(video=args.video, path=args.path, skip_cog=args.skip_cog, skip_track=args.skip_track,
              plt_graph=args.plt_graph, ratio_band=args.ratio_band, graph_num=args.graph_num,
              cog_color=args.cog_color, start_frame=args.start_frame, debug=args.debug)

