import torch
import pandas as pd
import os
from PIL import Image
from model import FoodComparator
from utils import get_transform

# --- ตั้งค่า Device และ Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FoodComparator()
model.load_state_dict(torch.load("model/food_model.pth", map_location=device))
model.to(device)
model.eval()

transform = get_transform(train=False)

def predict(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img1, img2)
        # ใช้ torch.sigmoid หรือ softmax ขึ้นอยู่กับโครงสร้าง model 
        # แต่จากโค้ดเดิมที่ใช้ argmax คาดว่าเป็น binary classification 2 class
        pred = torch.argmax(output, dim=1) 
    return pred.item() + 1

if __name__ == "__main__":
    csv_path = "testset1/test1.csv"
    image_folder = "testset1/Test Images1"
    output_path = "testset1/predictions1.csv" # <--- ชื่อไฟล์ผลลัพธ์
    
    df = pd.read_csv(csv_path)
    predictions = [] # ลิสต์สำหรับเก็บผลทำนาย

    print(f"กำลังประมวลผลทั้งหมด {len(df)} คู่...")

    for index, row in df.iterrows():
        img1_full_path = os.path.join(image_folder, row["Image 1"])
        img2_full_path = os.path.join(image_folder, row["Image 2"])
        
        try:
            result = predict(img1_full_path, img2_full_path)
            predictions.append(result)
            if (index + 1) % 10 == 0: # แสดง Progress ทุกๆ 10 แถว
                print(f"ทำนายเสร็จแล้ว {index + 1}/{len(df)} คู่")
        except Exception as e:
            print(f"Error ที่แถว {index}: {e}")
            predictions.append("Error")

    # เพิ่มคอลัมน์ผลลัพธ์เข้าไปใน DataFrame
    df["Winner"] = predictions

    # บันทึกเป็น CSV
    df.to_csv(output_path, index=False)
    print("-" * 30)
    print(f"เสร็จสมบูรณ์! บันทึกไฟล์ไปที่: {output_path}")