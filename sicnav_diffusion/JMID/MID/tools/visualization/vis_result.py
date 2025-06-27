import argparse
import pandas as pd
import os
import cv2
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of MID")
    parser.add_argument(
        "--ann_data", default="./pixel_raw_data/pixel_eth/train/crowds_zara01_train.txt"
    )
    parser.add_argument("--video_data", default="./videos/crowds_zara01.avi")
    parser.add_argument("--output_dir", default="./vis_results")
    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_data)
    if not cap.isOpened():
        sys.exit()

    data = pd.read_csv(args.ann_data, sep="\t", index_col=False, header=None)
    data.columns = ["frame_id", "track_id", "pos_x", "pos_y"]

    iter = 0
    while True:
        ret, frame = cap.read()
        if ret:
            sample_data = data[data["frame_id"] == iter]
            if len(sample_data) > 0:
                for index, row in sample_data.iterrows():
                    print("{} {}".format(row["pos_x"], row["pos_y"]))
                    cv2.circle(
                        frame,
                        center=(int(row["pos_x"]), int(row["pos_y"])),
                        radius=5,
                        color=(0, 255, 0),
                        thickness=-1,
                    )
                cv2.imwrite(
                    os.path.join(args.output_dir, "{:0>6}.jpg".format(iter)), frame
                )
            iter += 1


if __name__ == "__main__":
    main()
