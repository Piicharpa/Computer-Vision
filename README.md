
# Computer Vision Portfolio: Contest & Exercise Tech

ยินดีต้อนรับสู่โปรเจกต์ **Computer Vision** ที่รวบรวมโซลูชันด้านการตรวจจับและวิเคราะห์ภาพ (Image & Video Analytics) โดยเน้นไปที่การใช้งานจริงในด้านการตรวจนับวัตถุและวิทยาศาสตร์การกีฬา

---

## Key Features

### 1. Coin Counting System
ระบบตรวจนับเหรียญอัตโนมัติจากภาพถ่าย โดยใช้เทคนิคการประมวลผลภาพเพื่อแยกแยะชนิดของเหรียญและคำนวณยอดเงินรวม
* **Tech Stack:** OpenCV, Image Processing
* **Capabilities:** Detects various coin sizes and sums the total currency value.

### 2. Squat Counting Analytics
ระบบวิเคราะห์ท่าทางและนับจำนวนการเล่น Squat ผ่านวิดีโอแบบ Real-time เพื่อช่วยในการออกกำลังกายอย่างถูกต้อง
* **Tech Stack:** MediaPipe, Pose Estimation, Python
* **Capabilities:** Track joints, calculate knee angles, and count repetitions automatically.

### 3. Contest Model Training
ส่วนของการพัฒนาโมเดลสำหรับการแข่งขัน (Contest) ซึ่งรวมไปถึงการจัดการ Dataset และการ Train โมเดลที่มีความแม่นยำสูง

---

## 📁 Project Structure

```text
.
├── CoinCounting/        # ระบบนับเหรียญและภาพตัวอย่าง
├── Contest/             # โปรเจกต์สำหรับการแข่งขัน (Training & Prediction)
│   ├── model.py         # โครงสร้างโมเดล
│   ├── train.py         # สคริปต์สำหรับเทรนโมเดล
│   └── requirements.txt # รายการ Library ที่จำเป็น
├── SquatCounting/       # ระบบวิเคราะห์ท่า Squat
└── README.md
