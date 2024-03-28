#from fastapi import FastAPI, File, UploadFile,Form,  Depends
#from xml.dom import WrongDocumentErr
from fastapi import FastAPI, File, UploadFile

#from typing import Optional
from PIL import Image
from io import BytesIO
#from pydantic import BaseModel#это понадобилось для втрого варианта
#нужно для работы с файлами с айфона
from pillow_heif import register_heif_opener

from datetime import datetime
import os

#import imghdr
#пробуем вставить логирование всех запросов, которые приходят на сервер
from typing import Callable
#from fastapi import APIRouter,  Request, Response
#from fastapi.routing import APIRoute
#import time
import gc
#import asyncio

#import magic #pip install python-magic
from tempfile import NamedTemporaryFile



register_heif_opener()

import prod #в этом  файле хранятся все процедуры

app = FastAPI()
#router = APIRouter(route_class=TimedRoute) #ЭТО ДЛЯ РОУТЕРА
#ПОЛУЧЕНИЕ ТОЛЬКО КАРТИНКИ. ОТВЕТ json
keys = ['hard', 'control', 'sexy']
wrong = {'need_moderation': 3, 'net1': {'hard': 0.0, 'control': 0.0, 'sexy': 0.0}, 'net2': {'hard': 0.0, 'control': 0.0, 'sexy': 0.0}}

IND = 0

@app.post("/ps")
async def analyze_image1(image: UploadFile = File(...)):
    global IND
    IND += 1
   
    ##if image_type or image.filename.lower().endswith('heic'):
    if True : #тут можно проверить изображение ли это вообще...см.строку выше
        with Image.open(BytesIO(await image.read())) as img:
          
            if img.mode != 'RGB':
                img = img.convert("RGB")
            pok = await  prod.look_to_file([img])#отправляем изображение и то, что требуется вернуть
        if IND >= 500:
            IND = 0
            #print('чисто')
            gc.collect() #почистимся 
        return pok[0] #у нас только 1 картинка, так что отдаем только первый ответ
    else:
         return None


#import ffmpeg

@app.post("/vd")
async def analyze_video(video: UploadFile = File(...)):
    global IND
    IND += 1

    filename = video.filename
    # Разделяем имя файла на имя и расширение
    name, extension = os.path.splitext(filename)
    #print(name, extension)
    
    video_file = await video.read()
    temp_file = NamedTemporaryFile(delete=False, suffix=extension)
    temp_file.write(video_file)

    #

    
    pok = await  prod.look_to_video_file(temp_file.name)
    #print(video.content_type)
    gc.collect() #почистимся 
    temp_file.close()
    os.remove(temp_file.name)
    return pok
    








@app.post("/test")
async def analyze_image(image: UploadFile = File(...)):
    with Image.open(BytesIO(await image.read())) as img:
         current_datetime = datetime.now()
         str_current_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
         filename = f"inp_{str_current_datetime}.jpg"
         filename = os.path.join('arh', filename)
         #img.save(filename)
         return {"width": img.width, "height": img.height}



txt = 'Этот адрес url не предназначен для открытия в браузере. Используйте post запрос в формате requests.post(url, files=file) где - file - это бинарный файл с изображением для распознавания'



@app.get("/ps")
async def not_use():
   
   
    return txt

@app.get("/test")
async def not_use():
   
    return txt

#app.include_router(router)
