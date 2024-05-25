import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import transforms 
import cv2
from PIL import Image
from torchvision import models 


test_transform =   transforms.Compose([
        transforms.Resize ( (150 , 150) ),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =  model = models.vgg19(pretrained = True)
model.classifier = nn.Sequential(
      
  nn.Linear(in_features=25088, out_features=2048) ,
  nn.ReLU(),
  nn.Linear(in_features=2048, out_features=512) ,
  nn.ReLU(),
  nn.Dropout(p=0.6), 
    
  nn.Linear(in_features=512 , out_features=3),
  nn.LogSoftmax(dim=1)  
)
model = model.to(device)
model.load_state_dict(torch.load('./vgg19_v1_97_3O.pth'))
model.eval()
videoCapture = cv2.VideoCapture(r'./animals.mp4')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

ps = 25
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
label_ref = {0:"Monkey", 1:"no Monkey",2:"object"}
x = 0 
y = 0 
a =0 
z = 0
with torch.no_grad():
    success, frame = videoCapture.read()
    while success:
        frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_copy = Image.fromarray(frame_copy)
        image_tensor = test_transform(frame_copy)
        image_tensor = image_tensor.unsqueeze(0) 
        test_input = image_tensor.to(device)
        outputs = model(test_input)
        _, predicted = torch.max(outputs, 1)
        probability =  F.softmax(outputs, dim=1)
        top_probability, top_class = probability.topk(1, dim=1)
        predicted = predicted.cpu().detach().numpy()
        predicted = predicted.tolist()[0]
        top_probability = top_probability.cpu().detach().numpy()
        top_probability = top_probability.tolist()[0][0]
        top_probability = '%.2f%%' % (top_probability * 100)
        print(top_probability)
        print(label_ref[predicted]) 
        z=z+1
        if(predicted == 1):
            x = x+1
        elif(predicted==0):
            y = y+1
        else:
            a = a+ 1
        
        frame = cv2.putText(frame, label_ref[predicted]+': '+top_probability, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        success, frame = videoCapture.read()
print("object")
print(a/z)
print("not monkey")
print(x/z)
print("monkey")
print(y/z)