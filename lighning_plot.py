import wandb
from model.network import Converter
import matplotlib.pyplot as plt
import os
import re

import numpy as np
import pandas as pd
from assets import AssetManager

from tqdm import tqdm
import cv2


def pred_imgs(converter, imgs):
    curr_imgs = np.stack(imgs, axis=0)
    content_codes = converter.content_encoder.predict(curr_imgs)
    class_codes = converter.class_encoder.predict(curr_imgs)
    class_adain_params = converter.class_modulation.predict(class_codes)
    return content_codes, class_adain_params


def ligning_plot(model_name):
    assets = AssetManager('results')
    converter = Converter.load(assets.get_model_dir(model_name), include_encoders=True)

    base_dir = r'data\small_norb_lord'

    azimuths = []
    elevations = []
    lightings = []
    lt_rts = []
    classes = []
    img_paths = []

    regex = re.compile(r'azimuth(\d+)_elevation(\d+)_lighting(\d+)_(\w+).jpg')
    for category in tqdm(os.listdir(base_dir)):
        for instance in os.listdir(os.path.join(base_dir, category)):
            for file_name in os.listdir(os.path.join(base_dir, category, instance)):
                img_path = os.path.join(base_dir, category, instance, file_name)
                azimuth, elevation, lighting, lt_rt = regex.match(file_name).groups()

                class_id = (int(category) * 10) + int(instance)
                azimuths.append(int(azimuth))
                elevations.append(int(elevation))
                lightings.append(int(lighting))
                lt_rts.append(lt_rt)
                classes.append(class_id)
                img_paths.append(img_path)

    df = pd.DataFrame({
        'azimuth': azimuths,
        'elevation': elevations,
        'lighting': lightings,
        'lt_rt': lt_rts,
        'classe': classes,
        'img_path': img_paths,
    })

    df = df.sample(frac=1).reset_index(drop=True)

    fxd_content = [df[df.lighting == i]['img_path'].iloc[0] for i in
                   range(6)]
    fxd_class = df[df.classe == 0]['img_path'][:10]
    l2li = lambda x: [
        np.expand_dims(cv2.cvtColor(cv2.resize(plt.imread(i), dsize=(64, 64)), cv2.COLOR_BGR2GRAY), axis=2).astype(
            np.float32) / 255.0 for i in x]

    fxd_content_img = l2li(fxd_content)
    fxd_class_img = l2li(fxd_class)

    fxd_content_cnt, fxd_content_cls = pred_imgs(converter, fxd_content_img)
    fxd_class_cnt, fxd_class_cls = pred_imgs(converter, fxd_class_img)

    plt.rcParams["figure.figsize"] = (20, 20)
    blank = np.zeros_like(fxd_content_img[0])
    output = [np.concatenate([blank] + list(fxd_content_img), axis=1)]
    for i in tqdm(range(10)):
        generated_imgs = [
            converter.generator.predict([fxd_content_cnt[[j]], fxd_class_cls[[i]]])[0]
            for j in range(6)
        ]

        converted_imgs = [fxd_class_img[i]] + generated_imgs

        output.append(np.concatenate(converted_imgs, axis=1))

    merged_img = np.concatenate(output, axis=0)

    plt.xlabel('Content')
    plt.ylabel('Class')
    plt.imshow(np.squeeze(merged_img), cmap='gray')
    wandb.log({"Lighting plot": plt})
