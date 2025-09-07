import cv2
import os
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits
from moviepy.editor import VideoFileClip
import numpy as np
from globals import xm_per_pix, time_window

# Initialize lane line objects and frame counter
processed_frames = 0
line_lt = Line(buffer_len=time_window)
line_rt = Line(buffer_len=time_window)


def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)
    off_x, off_y = 20, 15

    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # thumbnails
    thumb_binary = cv2.resize(img_binary, (thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary]*3) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w] = thumb_binary

    thumb_birdeye = cv2.resize(img_birdeye, (thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye]*3) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w)] = thumb_birdeye

    thumb_img_fit = cv2.resize(img_fit, (thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w)] = thumb_img_fit

    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, f'Curvature radius: {mean_curvature_meter:.2f} m', (860, 60), font, 0.9, (255,255,255), 2)
    cv2.putText(blend_on_road, f'Offset from center: {offset_meter:.2f} m', (860, 130), font, 0.9, (255,255,255), 2)

    return blend_on_road


def compute_offset_from_center(line_lt, line_rt, frame_width):
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = (line_lt_bottom + lane_width / 2) - midpoint
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1
    return offset_meter


def process_pipeline(frame, keep_state=True):
    global line_lt, line_rt, processed_frames

    img_undistorted = undistort(frame, mtx, dist)
    img_binary = binarize(img_undistorted)
    img_birdeye, M, Minv = birdeye(img_binary)

    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt)

    offset_meter = compute_offset_from_center(line_lt, line_rt, frame.shape[1])
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)

    processed_frames += 1
    return blend_output


if __name__ == '__main__':
    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    # Ask user for video file
    video_path = input("Enter path to the video file: ").strip()
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"{video_path} not found")

    # Process the video
    clip = VideoFileClip(video_path).fl_image(process_pipeline)
    output_video = f"output_{os.path.basename(video_path)}"
    clip.write_videofile(output_video, audio=False)
    print(f"Processed video saved as {output_video}")


# import cv2
# import numpy as np
# from calibration_utils import calibrate_camera, undistort
# from binarization_utils import binarize
# from perspective_utils import birdeye
# from line_utils import get_fits_by_sliding_windows, get_fits_by_previous_fits, draw_back_onto_the_road, Line
# from globals import xm_per_pix, time_window

# # Initialize lane line objects and frame counter
# processed_frames = 0
# line_lt = Line(buffer_len=time_window)
# line_rt = Line(buffer_len=time_window)

# # Camera calibration
# ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')


# def compute_offset_from_center(line_lt, line_rt, frame_width):
#     if line_lt.detected and line_rt.detected:
#         line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
#         line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
#         lane_width = line_rt_bottom - line_lt_bottom
#         midpoint = frame_width / 2
#         offset_pix = (line_lt_bottom + lane_width / 2) - midpoint
#         offset_meter = xm_per_pix * offset_pix
#     else:
#         offset_meter = -1
#     return offset_meter


# def process_pipeline(frame, keep_state=True):
#     global line_lt, line_rt, processed_frames

#     img_undistorted = undistort(frame, mtx, dist)
#     img_binary = binarize(img_undistorted)
#     img_birdeye, M, Minv = birdeye(img_binary)

#     # Use previous fit if available, otherwise sliding windows
#     if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
#         line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt)
#     else:
#         line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt)

#     # Fallback if fit failed
#     if line_lt.last_fit_pixel is None:
#         line_lt.last_fit_pixel = np.array([0, 0, 0])
#     if line_rt.last_fit_pixel is None:
#         line_rt.last_fit_pixel = np.array([0, 0, 0])

#     offset_meter = compute_offset_from_center(line_lt, line_rt, frame.shape[1])
#     blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)

#     processed_frames += 1
#     return blend_on_road


# if __name__ == '__main__':
#     cap = cv2.VideoCapture(0)  # Use default webcam

#     print("Press 'q' to quit the live lane detection.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Resize frame for faster processing (optional)
#         frame = cv2.resize(frame, (640, 360))

#         output_frame = process_pipeline(frame, keep_state=True)
#         cv2.imshow("Lane Detection", output_frame)

#         # Press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
