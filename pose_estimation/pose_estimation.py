import cv2
import numpy as np
import pandas as pd
import os

import ultralytics
import torch

# model = ultralytics.YOLO(model='yolov8m-pose.pt')

def pose_estimation_video(model, video_path: str, csv_path: str):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    results = []
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            result = model(frame)
            result = result[0].cpu()

            # Save relevent data
            kpts = np.concatenate((result.keypoints.numpy()[0].data, result.keypoints.numpy()[0].xyn), axis=2)
            kpts = np.reshape(kpts, (1, np.product(kpts.shape)))
            boxe = np.concatenate((result.boxes.numpy()[0].conf, result.boxes.numpy()[0].xywh[0], result.boxes.numpy()[0].xywhn[0]), axis=0)
            total = np.concatenate((boxe, kpts[0]), axis=0)

            results.append(total)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    df = pd.DataFrame(results)
    header = ["box_conf", "box_x", "box_y", "box_w", "box_h", "box_x_n", "box_y_n", "box_w_n", "box_h_n",
                "1_x", "1_y", "1_conf", "1_x_n", "1_y_n",
                "2_x", "2_y", "2_conf", "2_x_n", "2_y_n",
                "3_x", "3_y", "3_conf", "3_x_n", "3_y_n",
                "4_x", "4_y", "4_conf", "4_x_n", "4_y_n",
                "5_x", "5_y", "5_conf", "5_x_n", "5_y_n",
                "6_x", "6_y", "6_conf", "6_x_n", "6_y_n",
                "7_x", "7_y", "7_conf", "7_x_n", "7_y_n",
                "8_x", "8_y", "8_conf", "8_x_n", "8_y_n",
                "9_x", "9_y", "9_conf", "9x_n", "9_y_n",
                "10_x", "10_y", "10_conf", "10_x_n", "10_y_n",
                "11_x", "11_y", "11_conf", "11_x_n", "11_y_n",
                "12_x", "12_y", "12_conf", "12_x_n", "12_y_n",
                "13_x", "13_y", "13_conf", "13_x_n", "13_y_n",
                "14_x", "14_y", "14_conf", "14_x_n", "14_y_n",
                "15_x", "15_y", "15_conf", "15_x_n", "15_y_n",
                "16_x", "16_y", "16_conf", "16_x_n", "16_y_n",
                "17_x", "17_y", "17_conf", "17_x_n", "17_y_n",
            ]
    df = df.set_axis(header,axis=1) # adding headers to the dataframe
    df.to_csv(csv_path, sep=';', decimal='.', index=False) # store keypoint csv

    # Release the video capture object and close the display window
    cap.release()

def pose_estimation_dataset(model, pathToData: str, pathToOutput: str):
    listeErreur = []

    for folder in sorted(os.listdir(pathToData)):
        if (not folder.startswith(".")) and os.path.isdir(os.path.join(pathToData, folder)):
            if folder != "test":
                pathToCSVFolder = os.path.join(pathToOutput, folder) # create a CSV folder for train and validation datasets
                if not os.path.exists(pathToCSVFolder):
                    os.mkdir(pathToCSVFolder)
                pathToFolder = os.path.join(pathToData, folder)
                for subFolder in sorted(os.listdir(pathToFolder)): 
                    if not subFolder.startswith(".") and os.path.isdir(os.path.join(pathToFolder, subFolder)):
                        pathToCSVSubFolder = os.path.join(pathToCSVFolder, subFolder) # create a CSV folder for each class of strokes
                        if not os.path.exists(pathToCSVSubFolder):
                            os.mkdir(pathToCSVSubFolder)
                        pathToSubFolder = os.path.join(pathToFolder, subFolder)
                        for file in sorted(os.listdir(pathToSubFolder)):
                            if (not file.startswith(".")) and (file.endswith(".mp4")):
                                pathToFile = os.path.join(pathToSubFolder, file)
                                CSVFileName = file[:len(file) - 4]
                                pathToCSV = os.path.join(pathToCSVSubFolder, CSVFileName+".csv")
                                if not os.path.exists(pathToCSV):
                                    print(pathToFile)
                                    try:
                                        pose_estimation_video(model, pathToFile, pathToCSV)
                                    except:
                                        print("Erreur fichier: "+pathToFile)
                                        listeErreur.append(pathToFile)
    return listeErreur