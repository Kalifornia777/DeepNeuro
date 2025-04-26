import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import shutil

# --- 1. Определение устройства для вычислений ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Работаем на устройстве: {device}")

# --- 2. Организация набора данных ---
def reorganize_dataset(directory):
    """Переименование файлов изображений для согласованности"""
    if not os.path.exists(directory):
        print(f"Папка не найдена: {directory}")
        return

    label = os.path.basename(directory)
    for idx, file in enumerate(os.listdir(directory)):
        path = os.path.join(directory, file)
        if os.path.isfile(path):
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                new_filename = f"{label}_{idx}{ext}"
                new_path = os.path.join(directory, new_filename)
                shutil.move(path, new_path)
                print(f"✔️ {file} --> {new_filename}")

# Проходим по всем классам и поднабору (train/test)
classes = ['city', 'money', 'rappers']
splits = ['train', 'test']

for split in splits:
    for cls in classes:
        reorganize_dataset(os.path.join('custom_dataset', split, cls))

# --- 3. Преобразования изображений ---
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Загрузка данных
data_root = os.path.abspath('custom_dataset')

train_data = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=transform_pipeline)
test_data = datasets.ImageFolder(root=os.path.join(data_root, 'test'), transform=transform_pipeline)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Количество обучающих изображений: {len(train_data)}")
print(f"Количество тестовых изображений: {len(test_data)}")

# --- 4. Инициализация модели ---
# Используем предобученную сеть DenseNet121
model = models.densenet121(pretrained=True)

# Замораживаем все слои кроме классификатора
for param in model.features.parameters():
    param.requires_grad = False

# Настраиваем последний слой под 3 класса
model.classifier = nn.Linear(model.classifier.in_features, 3)
model = model.to(device)

# --- 5. Параметры обучения ---
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# --- 6. Процесс обучения ---
print("\n🚀 Обучение начинается!")

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Эпоха [{epoch+1}/{num_epochs}], Средняя потеря: {avg_loss:.4f}")

# --- 7. Тестирование модели ---
print("\n🧪 Оценка модели на тестовом наборе...")

model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_predictions += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()

accuracy = (correct_predictions / total_predictions) * 100
print(f"Точность на тестовых данных: {accuracy:.2f}%")

# --- 8. Сохраняем обученную модель ---
model_save_path = "densenet121_city_money_rappers.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✅ Модель сохранена как '{model_save_path}'")
