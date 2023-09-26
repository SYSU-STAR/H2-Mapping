import cv2
from tqdm import tqdm


def make_video(file_path, output_path, fps, size, total_frame):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    output_movie_color = cv2.VideoWriter(output_path + "/color_rendering.mp4", fourcc, fps, size)

    frame_number = 0
    for i in tqdm(range(0, total_frame)):
        file_name_color = "frame{:06d}.jpg".format(i)
        frame_color = cv2.imread(file_path + "/mapping_vis/color/" + file_name_color, cv2.IMREAD_UNCHANGED)

        frame_number += 1
        output_movie_color.write(frame_color)

    # All done!
    cv2.destroyAllWindows()
