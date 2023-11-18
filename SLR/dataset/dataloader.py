import os
import torch
import argparse
import cv2 as cv
import numpy as np
import torchvision
import mediapipe as mp
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

class VideoDataSet(torch.utils.data.Dataset):

    def __init__(self, folder_root: str, num_frames: int, data_name: str, split: str, image_size: int) -> None:

        self.folder_root = folder_root
        self.num_frames = num_frames
        self.data_name = data_name
        self.image_size = image_size
        self.split = split

        self.src_path = os.path.join(self.folder_root, self.data_name)
        self.dst_path = os.path.join(self.folder_root, "preprocessing")
        self.folder2labelint = os.path.join(self.folder_root, "folder2label_int.txt")
        self.data_folder = os.path.join(self.dst_path, self.split)


        self.preprocessing_all_video()


        folder2label = {}
        with open(self.folder2labelint, 'r') as file:
            data = file.read()
            data = data.split("\n")
            for i in data:
                i = i.split(" ")
                folder2label[i[0]] = int(i[1])
        

        self.labels = []
        self.poses_folder = []
        self.frames_folder = []

        for folder in os.listdir(self.data_folder):
            label_folder = os.path.join(self.data_folder, folder)
            for label in os.listdir(label_folder):
                item_folder = os.path.join(label_folder, label)
                for item in os.listdir(item_folder):
                    
                    if folder == "pose":
                        self.poses_folder.append(os.path.join(item_folder, item))
                    else:
                        self.frames_folder.append(os.path.join(item_folder, item))
                    self.labels.append(folder2label[label])

    def preprocessing_all_video(self) -> None:

        if not os.path.exists(self.dst_path):
            print("This preprocessing process will take a long time but only happens once !!!")
            self.data_split()
            print("Complete preprocessing !!!")
    

    def data_split(self):

        folder_train = os.path.join(self.dst_path, "train")
        folder_test = os.path.join(self.dst_path, "test")

        os.makedirs(self.dst_path)
        os.makedirs(folder_train)
        os.makedirs(folder_test)

        all_videos, all_labels = [], []
        for label in os.listdir(self.src_path):
            label_folder = os.path.join(self.src_path, label)
            for video_name  in os.listdir(label_folder):
                video_path = os.path.join(label_folder, video_name)
                all_labels.append(label)
                all_videos.append((video_path, video_name[:-4]))
        
        all_videos = np.array(all_videos)
        all_labels = np.array(all_labels)

        print("Data Split")

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for train_index, test_index in sss.split(all_videos, all_labels):
            videos_train, videos_test = all_videos[train_index], all_videos[test_index]
            labels_train, labels_test = all_labels[train_index], all_labels[test_index]

        #Train;
        self.preprocessing_folder(folder_train, videos_train, labels_train)

        #Test;
        self.preprocessing_folder(folder_test, videos_test, labels_test)

        
    def preprocessing_folder(self, dst_folder_path: str, all_videos: list, all_labels: list) -> None:

        folder_frames = os.path.join(dst_folder_path, "frames")
        folder_pose = os.path.join(dst_folder_path, "pose")

        os.makedirs(folder_frames)
        os.makedirs(folder_pose)

        for index in range(len(all_videos)):
            video_path = all_videos[index][0]
            video_name = all_videos[index][1]
            label = all_labels[index]

            if self.check_video(video_path):
                self.get_frames(video_path, label, video_name,folder_frames, folder_pose)
            else:
                print(f"Can't Open Video {video_path}")
            

    def get_frames(self, video_path: str, label: str, name:str, dst_frames: str, dst_pose: str):

        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        new_dst_frames = os.path.join(dst_frames, label)
        new_dst_frames = os.path.join(new_dst_frames, name)

        new_dst_pose = os.path.join(dst_pose, label)
        new_dst_pose = os.path.join(new_dst_pose, name)
        
        os.makedirs(new_dst_frames)
        os.makedirs(new_dst_pose)

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
                mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:
            
            cap = cv.VideoCapture(video_path)

            count_frame = 0
            num_count_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            selected_frames = [int(i * (num_count_frame - 1) / (self.num_frames - 1)) for i in range(self.num_frames)]

            for frame_index in selected_frames:

                cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
                success, image = cap.read()
                image_tmp = image.copy()

                image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                w, h, c = image_rgb.shape
                image_black = np.zeros((w, h, c), dtype=np.uint8)

                face_results = face_detection.process(image_rgb)
                if face_results.detections:
                    for detection in face_results.detections:
                        mp_drawing.draw_detection(image, detection)

                pose_results = pose_detection.process(image_rgb)
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(image_black, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    connections = [
                        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
                        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER)]

                    for connection in connections:
                        start_point = pose_results.pose_landmarks.landmark[connection[0]]
                        end_point = pose_results.pose_landmarks.landmark[connection[1]]
                        start_x, start_y = int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0])
                        end_x, end_y = int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0])
                        cv.line(image_black, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

                image_name = label + "_" + str(count_frame) + ".jpg"
                count_frame += 1

                folder_frames_idx = os.path.join(new_dst_frames, image_name)
                folder_pose_idx = os.path.join(new_dst_pose, image_name)

                cv.imwrite(folder_frames_idx, image_tmp)
                cv.imwrite(folder_pose_idx, image_black)

    def check_video(self, video_path: str) -> int:
        cap = cv.VideoCapture(video_path)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not cap.isOpened() and frame_count < self.num_frames:
            return 0
        return 1

    def __len__(self) -> int:

        return len(self.frames_folder)

    def __getitem__(self, index: int):

        frame_path = self.frames_folder[index]
        pose_path = self.poses_folder[index]
        label = self.labels[index]

        frames, poses = self.putitem(frame_path, pose_path)

        if self.split == "train":

            transform = None

            if np.random.randint(2) == 1:
                angle = int(np.random.uniform(0, 45))

                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(degrees=(angle, angle)),
                    transforms.CenterCrop((190, 190)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            else:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            frames = frames.permute(0, 3, 1, 2)
            for i in range(frames.size(0)):
                frames[i] = transform(frames[i])
            frames = frames.permute(1, 0, 2, 3)

            poses = poses.permute(0, 3, 1, 2)
            for i in range(poses.size(0)):
                poses[i] = transform(poses[i])
            poses = poses.permute(1, 0, 2, 3) 

        return frames, poses, label
    
    
    def putitem(self, frame_path: str, pose_path) -> torch.tensor:
        
        frames = torch.empty(self.num_frames, self.image_size, self.image_size, 3)
        posese = torch.empty(self.num_frames, self.image_size, self.image_size, 3)

        frames_lst = sorted(os.listdir(frame_path), key=extract_frame_number)
        poses_lst = sorted(os.listdir(pose_path), key=extract_frame_number)

        count = 0
        for idx_frame in frames_lst:

            frame_path_tmp = os.path.join(frame_path, idx_frame)
            frame = cv.imread(frame_path_tmp)
            frame = cv.resize(frame, (self.image_size, self.image_size))
            frame = torch.tensor(frame, dtype=torch.float32) / 255.0
            frames[count] = frame
            count += 1

        count = 0
        for idx_pose in poses_lst:
            pose_path_tmp = os.path.join(pose_path, idx_pose)
            pose = cv.imread(pose_path_tmp)
            pose = cv.resize(pose, (self.image_size, self.image_size))
            pose = torch.tensor(pose, dtype=torch.float32) / 255.0
            posese[count] = pose
            count += 1
        
        return frames, posese

def extract_frame_number(file_name):
    try:
        return int(file_name.split('_')[1].split('.')[0])
    except (IndexError, ValueError):
        return float('inf')
    

def test(tensor_frames, path):

    tensor_frames = tensor_frames.permute(1, 2, 3, 0)
    for idx_frame in range(tensor_frames.size()[0]):
        image = tensor_frames[idx_frame].numpy()
        image = (image * 255).astype(np.uint8)
        name = "image" + str(idx_frame) + ".png"
        image_name = os.path.join(path, name)
        cv.imwrite(image_name, image)
    
    print("Done")

# if __name__ == "__main__":
#     train_loader = torch.utils.data.DataLoader(VideoDataSet(folder_root=r"E:\dataset\dataset_wlasl100", num_frames=16, 
#                                                             data_name="WLASL100", split="train", image_size=224), 
#                                                             batch_size=1, shuffle=True, num_workers=4)
#     a, b, c = next(iter(train_loader))
#     print(c[0])
#     test(a[0], r"E:\rgb")
#     test(b[0], r"E:\pose")
