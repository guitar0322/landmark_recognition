import sys
import json
import numpy as np
import os
import matplotlib.pyplot as plt
# import cv2
from sklearn.decomposition import PCA
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model

get_ipython().run_line_magic('matplotlib', 'inline')

image_dir_basepath = '../data/images/'
image_test_basepath = '../data/images/Test'
names = ['Lottetower', 'Kyeongbok', '63building', 'Namsan', 'LeesunsinStatue']#임베딩할 랜드마크의 목록들.
image_size = 160

model_path = 'model/facenet_keras.h5'#모델 저장 경로를 로컬 환경에 맞게 변경하세요.
model = load_model(model_path)

def prewhiten(x):#이미지 미백작업
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10): #L2 norm 정규화.
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def resize_images(filepaths):#pretrained 모델에 맞는 인풋 사이즈로 resize
    images = []
    for filepath in filepaths:
        img = imread(filepath)
        img = np.array(img)
        img = resize(img, (160,160))
        images.append(img)
        
    return np.array(images)

def resize_image(image):
    img = np.array(image)
    img = resize(img, (160,160))
    return img

def calc_embs(filepaths, margin=10, batch_size=1):#landmark image의 embedding값 계산
    images = resize_images(filepaths)
    aligned_images = images
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))#모델 결과산출
    embs = l2_normalize(np.concatenate(pd))#정규화적용

    return embs

def calc_emb(image):
    resize_img = [resize_image(image)]
    resize_img = np.array(resize_img)
    emb = l2_normalize(model.predict_on_batch(resize_img))
    
    return emb

#임베딩결과를 json파일에 저장
def emb_to_json():
    data = {}
    for name in names:
        image_dirpath = image_dir_basepath + name
        image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]
        embs = calc_embs(image_filepaths)
        for i in range(len(image_filepaths)):
            data['{}{}'.format(name, i)] = {'name' : name,#딕셔너리에 embedding값 저장
                                            'emb' : embs[i].tolist()}
    
    with open('../data/jsondata.json', 'w', encoding='utf-8') as json_file:
        json.dump(json.dumps(data), json_file, indent="\t")

def load_json():#json파일에 저장된 랜드마크의 인코딩벡터를 딕셔너리에 로딩.
    dic = {}
    with open('../data/jsondata.json') as json_data: #jsondata파일 경로 수정할것
        dic = json.load(json_data)
        dic = json.loads(dic)
    return dic

def calc_dist_dic(img_name0, img_name1, dic_data):#euclidean distance 알고리즘 dictionary를 활용
    return distance.euclidean(dic_data[img_name0]['emb'], dic_data[img_name1]['emb'])

def calc_dist(target_emd, dest_emd):#단일벡터 사용.
    return distance.euclidean(target_emd, dest_emd)


def recognition_landmark(target_img, emb_dic):
    target_emb = calc_emb(target_img)
    max_value = 1
    cur_value = 0
    result = {}
    for emb in emb_dic:
        cur_value = calc_dist(target_emb, emb_dic.get(emb)['emb'])
        if(max_value >= cur_value):
            max_value = cur_value
            result = emb_dic.get(emb)
    return result['name']

def main(target_image_path):
    data = load_json()
    recognition_landmark(imread(target_image_path), data)
    
if __name__ == '__main__':
    main(sys.argv[1])

