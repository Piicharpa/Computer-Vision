import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from utils import get_transform

class FoodDataset(Dataset):
    #ส่วนนี้จะทำงาน ครั้งเดียว ตอนที่เราสร้าง Object เพื่อรวบรวมรายชื่อไฟล์ทั้งหมด
    def __init__(self, csv_paths, image_folder_root, transform=None, train=True):
        dfs = []
        for path in csv_paths:
            df = pd.read_csv(path)
            dfs.append(df)
        self.data = pd.concat(dfs, ignore_index=True)
        #โหลด CSV: รับ List ของไฟล์ CSV มาวนลูปอ่าน แล้วนำมารวมกันเป็นตารางเดียว (pd.concat) เพื่อให้จัดการง่าย

        #กำหนด Transform: ตรวจสอบว่าถ้าเราส่ง transform มาจากหน้า train.py ก็ให้ใช้ตัวนั้น แต่ถ้าไม่ส่งมา ให้ใช้ค่าเริ่มต้นจาก get_transform(train)
        self.image_folder_root = image_folder_root
        self.transform = transform if transform else get_transform(train)
        
        # สร้าง "แผนที่" ของไฟล์ทั้งหมดที่มีอยู่ในโฟลเดอร์ เพื่อให้หาไฟล์ได้เร็วขึ้น
        self.image_path_map = {}
        for root, dirs, files in os.walk(self.image_folder_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # เก็บชื่อไฟล์เป็น Key และ Path เต็มเป็น Value
                    self.image_path_map[file] = os.path.join(root, file)

    def __len__(self):
        return len(self.data)

    def find_path(self, filename):
        # ถ้าใน CSV มีแค่ชื่อไฟล์ (เช่น b1_1.jpg) ให้ดึงจากแผนที่
        # แต่ถ้าใน CSV มี path บางส่วน ให้ลองหาแบบตรงตัวก่อน
        if filename in self.image_path_map:
            return self.image_path_map[filename]        
        # กรณีสำรอง: ถ้าหาไม่เจอในแผนที่ ให้ลองเอา root ไปต่อตรงๆ
        return os.path.join(self.image_folder_root, filename)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ใช้ฟังก์ชัน find_path ช่วยหาที่อยู่จริงของไฟล์
        img1_path = self.find_path(row["Image 1"])
        img2_path = self.find_path(row["Image 2"])

        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error: ค้นหาไฟล์ไม่เจอที่ {img1_path} หรือ {img2_path}")
            raise e

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        label = row["Winner"] - 1

        return img1, img2, label