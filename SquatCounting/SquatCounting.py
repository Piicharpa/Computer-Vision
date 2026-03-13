import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

# SETTINGS
TRACK_COLOR_MAP = {
    0: (0, 255, 255),
    1: (0, 255, 255),
    2: (0, 255, 255)
}

MOTION_THRESHOLD = 40



#   Graph
def extract_squat_repetitions(motion_trace, show_plot=False):
    squat_counts = []
    total_subjects = int(np.max(motion_trace[:, 1]) + 1)

    if show_plot:
        plt.figure(figsize=(10, 6))

    for sid in range(total_subjects):
        subject_data = motion_trace[motion_trace[:, 1] == sid]
        height_signal = subject_data[:, 5]

        peak_threshold = np.percentile(height_signal, 80)
        peaks, _ = find_peaks(height_signal, height=peak_threshold, distance=28)

        squat_counts.append(len(peaks))

    return squat_counts


#   FOREGROUND 
def foreground_activity_mask(frame, reference, blur_kernel, dilate_kernel):
    diff = cv2.absdiff(frame, reference)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(gray, MOTION_THRESHOLD, 255)
    mask = cv2.medianBlur(mask, blur_kernel)
    mask = cv2.dilate(mask, dilate_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 20), np.uint8))

    return mask


#   WALL
def track_wall_side_subject(view, reference, motion_trace, frame_no):
    mask = foreground_activity_mask(
        view, reference,
        blur_kernel=19,
        dilate_kernel=np.ones((65, 20), np.uint8)
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output = view.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 2 and w * h > 1e4 and frame_no >= 250:
            subject_id = 0
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                TRACK_COLOR_MAP[subject_id],
                1
            )
            motion_trace.append([
                frame_no, subject_id, x, y, w,
                view.shape[0] / h
            ])

    return output


#   DOOR 
def track_door_side_subjects(view, reference, motion_trace, frame_no):
    mask = foreground_activity_mask(
        view, reference,
        blur_kernel=17,
        dilate_kernel=np.ones((99, 19), np.uint8)
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output = view.copy()

    subject_id = 1
    for cnt in contours:
        if subject_id > 2:
            break

        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 2 and w * h > 8e3 and 300 <= frame_no <= 970:
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                TRACK_COLOR_MAP[subject_id],
                1
            )
            motion_trace.append([
                frame_no, subject_id, x, y, w,
                view.shape[0] / h
            ])
            subject_id += 1

    return output



#   MAIN 
def run_squat_evaluation(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, first_frame = cap.read()

    if not ok:
        print("❌ Video cannot be opened")
        return []

    wall_reference = first_frame[:, 110:305].copy()
    door_reference = first_frame[:, 310:490].copy()

    frame_no = 0
    motion_trace = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        wall_view = frame[:, 110:305].copy()
        door_view = frame[:, 310:490].copy()

        vis_wall = track_wall_side_subject(
            wall_view, wall_reference, motion_trace, frame_no
        )
        vis_door = track_door_side_subjects(
            door_view, door_reference, motion_trace, frame_no
        )

        cv2.imshow("Wall", vis_wall)
        cv2.imshow("Door", vis_door)

        frame_no += 1
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    motion_trace = np.array(motion_trace)

    # normalize
    motion_trace[:, 5] = (
        motion_trace[:, 5] - motion_trace[:, 5].min()
    ) / (
        motion_trace[:, 5].max() - motion_trace[:, 5].min()
    )

    # smooth + binarize per subject
    for sid in np.unique(motion_trace[:, 1]):
        idx = motion_trace[:, 1] == sid
        smoothed = gaussian_filter(motion_trace[idx, 5], sigma=3.5)
        motion_trace[idx, 5] = (smoothed >= np.mean(smoothed) - 0.01).astype(int)

    return extract_squat_repetitions(motion_trace, show_plot=False)


#   RUN TEST
print(run_squat_evaluation("./Squat1_8_9.avi"))
print(run_squat_evaluation("./Squat2_16_17.avi"))
print(run_squat_evaluation("./Squat3_11_9_10.avi"))
