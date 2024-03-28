
import torch
import asyncio

from torch import nn


import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from torchvision import transforms
from torchvision.transforms import v2
import cv2



#import ffmpeg  #нужно будет только если у нас будет ручное определение поворота видео
from PIL import Image  #это используется только 1 раз. но как избежать - не ясно. из numpy массива преобразуется в image 
import numpy as np


from fastapi.responses import FileResponse
import json

#import os
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

#МОДЕЛЬ 1.
#small 
#обучена на 20000 на класса  256x256
#точность 96.63 минимальный лосс 0.10854
#классификатор
#self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 32), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(32, 3)) #nn.Dropout(0.2),

#модель 2.
#small 
#обучена на 3000 на класса  320х320
#точность 95.06 минимальный лосс 0.15
#классификатор
#self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 32), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(32, 3)) #nn.Dropout(0.2),

#модель 3.
#BASE 
#обучена на 20000 на класса  266х266
#точность 97.09 минимальный лосс 0.08456  
#классификатор
#self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 256), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(256, len(class_names))) 





device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#device = 'cpu'
print('ВНИМАНИЕ!!! Обрабатываеться будет всё на ',device, '<<<<<<<<<<<<<----------------')


cls = {0:'hard',1 :'control', 2:'sexy'}
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')



class ResizeAndPad:
    def __init__(self, target_size, multiple):
        self.target_size = target_size
        self.multiple = multiple

    def __call__(self, img):
        # Resize the image
        img = v2.Resize(self.target_size)(img)

        # Calculate padding
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple

        # Apply padding
        img = v2.Pad((pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))(img)

        return img





from dinov2.models.vision_transformer import vit_small, vit_base #vit_large, vit_giant2
class DinoVisionTransformerClassifier(nn.Module):

    def __init__(self, model_size="small"):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.model_size = model_size

        # loading a model with registers
        n_register_tokens = 4

        if model_size == "small":
            self.transformer = vit_small(patch_size=14,
                              img_size=526,
                              init_values=1.0,
                              num_register_tokens=n_register_tokens,
                              block_chunks=0)
            self.embedding_size = 384
            self.number_of_heads = 6
            self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 32), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(32, 3)) #nn.Dropout(0.2),
            self.preprocessing = transforms.Compose([ ResizeAndPad(256, 14),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                             ]
                                            )




        elif model_size == "base":
            self.transformer = vit_base(patch_size=14,
                             img_size=526,
                             init_values=1.0,
                             num_register_tokens=n_register_tokens,
                             block_chunks=0,
                            #drop_path_rate=0.05
                            )
            self.embedding_size = 768
            self.number_of_heads = 12
            
            self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 256), nn.ReLU(), nn.Dropout(0.1),  nn.Linear(256, 3))
            
            #self.preprocessing = transforms.Compose([ ResizeAndPad(266, 14),
            #                                   transforms.ToTensor(),
            #                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #                                 ]
            #                                )

        



    def forward(self, imgs):
        tensor_imgs = [torch.from_numpy(np.array(img)).float() for img in imgs]
        x = torch.stack(tensor_imgs)
        x = x.to(device)
  
        
        with  torch.no_grad():
            x = self.transformer(x)
            x = self.transformer.norm(x)
            x = self.classifier(x)
        

        rez  = torch.softmax(x, dim=1)
        return rez


model1 = DinoVisionTransformerClassifier("small").to(device)
model2 = DinoVisionTransformerClassifier("base").to(device)


model1.load_state_dict(torch.load('model1.pth', map_location=torch.device(device)))
model2.load_state_dict(torch.load('model3.pth', map_location=torch.device(device)))


model1.eval()
model2.eval()




async def get_predict_from_model(img, model):
    probabilities = model.forward(img)
        
    #rez = dict(zip(cls.values(), probabilities.flatten().tolist()))
    rez = [dict(zip(cls.values(),r))  for r in probabilities.tolist()]
    return  rez




def get_predict_from_model_SIMPLE(img, model):
    probabilities = model.forward(img)
    
    rez = [dict(zip(cls.values(),r))  for r in probabilities.tolist()]
    return  rez

def prep(preprocessing, img):
        x = [preprocessing(i) for i in img]
        return x

prep1 =transforms.Compose([ ResizeAndPad(256, 14),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                             ]
                                            )

prep2 =transforms.Compose([ ResizeAndPad(266, 14),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                             ]
                                            )



#ФУНКЦИЯ ОБРАБОТКИ ИЗОБРАЖЕНИЙ
async def look_to_file(img):
    im1=prep(prep1,img)
    im2=prep(prep2,img)


    #ВАРИАНТ 1. АСИНХОННОСТЬ ЧЕРЕЗ ТАСКУ
    #task1 = asyncio.create_task(get_predict_from_model(img, model1))
    #task2 = asyncio.create_task(get_predict_from_model(img, model2))
    #r1 = await task1
    #r2 = await task2
    
    #ВАРИАНТ 2. АССИНХРОННОСТЬ ЧЕРЕЗ ГАДА
    r1, r2 = await asyncio.gather(
       get_predict_from_model(im1, model1),
       get_predict_from_model(im2, model2)
    )

    #ВАРИАНТ 3. ВООБЩЕ БЕЗ АСИНХРОННОСТИ
    #r1 = get_predict_from_model_SIMPLE(img, model)
    #r2 = get_predict_from_model_SIMPLE(img, model2)

    #print('r1', r1)
    #print('r2', r2)

    
    rez =[ dict([('need_moderation', 0 if (r1[i]['sexy'] >  0.85) & (r2[i]['sexy'] >  0.85) else  1),('net1', r1[i]), ('net2', r2[i])]) for i in range(len(r1))]
    
    return rez


#ВСЕ ЧТО ДАЛЕЕ ПРО ВИДЕО
BATCH_SIZE = 128  #сколько кадров можно положить одним куском на GPU
STEP_FRAME = 10   #будем обрабатывать каждый такой кадр. т.е. один из STEP_FRAME. Все обрабатывать не нужно. слишком ресурсоемко
                    #если поставить 1 - будут анализироваться ВСЕ КАДРЫ видео. Долго, но надежно. 
TMP_FILE ='output.jpg'  #это временный файл, который нам нужен, сюда попадет самый плохой кадр.


#def check_rotation(file):
#    #на входе имя файла
#    #проверяем не повернут ли файл. если повернут, придется вращать каждый файл
#    #данные о повороте видео хранятся в данных самого видео
#    #если данных нет, то скорей всего вращать не надо, возвращяем None.
#    #было бы супер, если видео приходили нормально повернуты, так как ffmpeg - какая то кривая. не всегда взлетает
#    #это запасная процедура, которая может использоваться если вдруг
#    try:
#        media_info = ffmpeg.probe(file)
#    except:
#        print('ВАЖНО! Похоже, что ffmpeg не установлен и система не может определить автоматически угол поворота видео. Установите в окружении')
#        print('с помощью команды pip install ffmpeg-python')
#        print('но она может не встать. Если не встала, смотри сюда: https://github.com/kkroening/ffmpeg-python/issues/392')
#        return 666
#    try:
#        u = media_info['streams'][0]['side_data_list'][0]['rotation']  #важно! вот этой штуковины, внутри медиа_инфо может и не быть, тогда вращать не надо
#        #print(u, type(u))
#        if u  == -90:
#            return cv2.ROTATE_90_CLOCKWISE
#        if u == 90:
#             return cv2.ROTATE_90_COUNTERCLOCKWISE
#        return None    #не надо вращать
#    except:
        
#        return None    #не надо вращать

def push_batch(batch):
    rez = look_to_file(batch)
    return rez

async def get_minimum_frame(list_of_frame, vidcap, rotateCode):
    #list_of_frame - список номеров кадров или итератор кадров, которые надо обработать
    #vidcap - Уже открытое видео для чтения кадров
    #как повернуть кадр
    batch = []     #бач изображений. Складываем сюда кадры по их индексу, бач потом закинем целиков в нейронку
    batch_ind = [] #номера кадров соответствующих бачу
    rating = []  #список для хранения результатов по каждому кадру
    for frame_number in list_of_frame:   
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  #устанавливаем указатель на нужное место
        _, image = vidcap.read()
        if rotateCode:                                #вращаем если надо
            image = cv2.rotate(image, rotateCode)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #приводим в нужную палитру.... получим numpy.ndarray
        img = Image.fromarray(image_rgb)                    #ВОЗВРАЩАЕМ В ИЗОБРАЖЕНИЕ
        
        

        #проверим не является ли это одноцветной заливкой. такие кадры не нужны, мы их просто пропускаем. Они плохо детектятся и их система отправляет на модерацию
        #img_array = np.array(img)
        #print(3, type(img_array))
        is_solid_color = np.all(image_rgb == image_rgb[0,  0])  #для проверки нужен np, по этому используем ранее полученный image_rgb
        if is_solid_color:
            continue


        #отправляем каждый кадр на проверку
        batch.append(img)       #здесь хранятся изображения
        batch_ind.append(frame_number) #а здесь их номера 

        if len(batch)==BATCH_SIZE:
                #накопили предельный размер бача - опустошаем, отправив на анализ
                
                rating.extend(zip(batch_ind, await push_batch(batch)))
                batch.clear()
                batch_ind.clear()
    if len(batch):
                #накопили - опустошаем
                rating.extend(zip(batch_ind,  await push_batch(batch)))
                batch.clear()
                batch_ind.clear()
    min_rating = 100 #столько быть не может, но ставим наверняка.
    frame_with_min_rating =''
    worst_result = ''
    #return 0, 0
    for ind_frame, rezult in rating:
        #print(ind_frame,rezult)
        if (rezult['net1']['sexy'] < min_rating) or (rezult['net2']['sexy'] < min_rating):
            min_rating=min(rezult['net1']['sexy'], rezult['net2']['sexy'])
            frame_with_min_rating = ind_frame
            worst_result = rezult
    return frame_with_min_rating, min_rating, worst_result #номер кадра с мин.рейтином, минимальный рейтинг, весь результат с мин.рейтином





async def look_to_video_file(temp_file):
  
    vidcap = cv2.VideoCapture(temp_file)
    vidcap.set(cv2.CAP_PROP_ORIENTATION_AUTO,  1)   #в документации написано, что это может работать не всегда...но в нашем случае работало всегда.  Далее есть возможность повернуть вручную 
    frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #rotateCode = check_rotation(temp_file) #процедура читает вручную надо ли повернуть видео....вдруг пригодится.
    rotateCode = None                       #



    first_list = range(0,frames, STEP_FRAME)  #выбираем кадры для первого прохода поиска кадра с минимальным рейтингом
    n,min_rating, worst_result =await get_minimum_frame(first_list,vidcap, rotateCode)  #отправляем список интересующихся номеров кадров, получаем номер кадра, значение 
    #print(1,n,min_rating,worst_result)
    
    if STEP_FRAME != 1:  #если шаг = 1 то второй проход делать не надо. и так уже найден минимум
        second_list = range(max(0,n - STEP_FRAME//2-1),min(frames-1, n + STEP_FRAME//2+1), 1)
        n,min_rating,worst_result =await get_minimum_frame(second_list,vidcap, rotateCode)
        #print(2, n,min_rating,worst_result)
    
    #получаем худший кадр
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, n)  #устанавливаем указатель на нужное место
    _, image = vidcap.read()
 
    cv2.imwrite(TMP_FILE,image)  #файл надо обязательно записать. без этого не получится его отправить в ответ
    
    file_response = FileResponse(TMP_FILE, filename="dangeros_image.jpg", media_type="image/jpeg")
    file_response.headers["result"] = json.dumps(worst_result)


    #os.remove(TMP_FILE)  #а удалять то его и нельзя... он нужен в return..но он каждый раз будет перезаписываться под этим именем

    return file_response #работает
  

