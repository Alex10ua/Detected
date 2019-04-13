from imageai.Detection import ObjectDetection
import os


exac_path=os.getcwd()#вказує шлях до цього проекту щоб програма знаходила додаткові файли


detector=ObjectDetection()
detector.setModelTypeAsRetinaNet()# встановлюємо те що використовуємо рітіна модель для визначення об єктів
detector.setModelPath(os.path.join(exac_path, "resnet50_coco_best_v2.0.1.h5")) #вказуємо шлях до моделі
detector.loadModel()# її завантаження

list=detector.detectObjectsFromImage(input_image=os.path.join(exac_path,"object2.jpg"), #надаємо методу зобразення
                                     output_image_path=os.path.join(exac_path,"Detected_objects.jpg") #видає проаналізоване зображення
                                     )