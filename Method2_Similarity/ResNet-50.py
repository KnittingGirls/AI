import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

model = models.resnet50(pretrained=True)
model.eval() 

model = nn.Sequential(*list(model.children())[:-1]) # 마지막 레이어 제거

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  
    with torch.no_grad():
        features = model(img_tensor)  
    return features.squeeze().numpy()

def calculate_similarity(img1_path, img2_path):
    features1 = extract_features(img1_path)
    features2 = extract_features(img2_path)
    similarity = cosine_similarity([features1], [features2])
    return similarity[0][0]

img1_path = "img/3.jpg"
img2_path = "img/2.jpeg"
similarity_score = calculate_similarity(img1_path, img2_path)
print(f"유사도: {similarity_score:.4f}")