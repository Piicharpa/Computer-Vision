import torch
from torch.utils.data import DataLoader, random_split
from dataset import FoodDataset
from model import FoodComparator
import os
import torchvision.transforms as transforms

# --- 1. เตรียมอุปกรณ์เหมือนเดิม ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# --- 2. โหลด Dataset และ "แบ่งข้อมูล" (สำคัญมาก) ---
full_dataset = FoodDataset(
    ["vote/data_from_questionaire.csv", "vote/data_from_intragram.csv"],
    "dataset",
    transform=transform
)

# แบ่ง Train 80% / Validation 20%
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# --- 3. ตั้งค่า Model ---
model = FoodComparator().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# --- 4. Loop การเทรนและการสอบ ---
epochs = 13
for epoch in range(epochs):
    # --- ช่วงเรียน (Training) ---
    model.train()
    train_loss = 0
    train_correct = 0
    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        pred = model(img1, img2)
        loss = criterion(pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += (pred.argmax(1) == label).sum().item()

    # --- ช่วงสอบ (Validation) ---
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            pred = model(img1, img2)
            val_correct += (pred.argmax(1) == label).sum().item()

    # --- แสดงผลลัพธ์ ---
    train_acc = (train_correct / train_size) * 100
    val_acc = (val_correct / val_size) * 100
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    print("-" * 30)

# --- 5. บันทึกผล ---
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/food_model.pth")
print("Model saved and tested!")