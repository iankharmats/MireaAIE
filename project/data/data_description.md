# Описание датасетов
   
## 1. **Humans** 
Открытый датасет, содержащий 3000+ лиц анфас и профиль, используется для детекции и параметризации. 

### Ссылка на скачивание: 
    https://www.kaggle.com/datasets/ashwingupta3012/human-faces/data


## 2. Cинтетический датасет шаржей
Состоит из ~300 пар «оригинал → шарж», сгенерированных с помощью предобученных моделей (ControlNet + Qwen-критик без дообучения), используется для дообучения ControlNet

### Ссылка на скачивание: 
    https://drive.google.com/drive/folders/1dOnPZoasui2zQFrwllEw6acHC1kmCCea?usp=sharing

### Разархивация
```bash
cd project/data
tar -xzvf dpo_dataset.tar.gz
```