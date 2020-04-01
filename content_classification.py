from model.network import Converter
import wandb
import numpy as np
from keras.callbacks import EarlyStopping
from assets import AssetManager

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


class LORDContentClassifier:
    def __init__(self, subset=None,
                 base_dir='results', model_name='minst_10_model', data_name='minst_10_test', include_encoders=True):
        assets = AssetManager(base_dir)
        data = np.load(assets.get_preprocess_file_path(data_name))
        imgs, classes, contents, n_classes = data['imgs'], data['classes'], data['contents'], data['n_classes']
        imgs = imgs.astype(np.float32) / 255.0
        if subset is not None:
            self.curr_imgs = imgs[:subset]
            self.classes = classes[:subset]
        else:
            self.curr_imgs = imgs
            self.classes = classes

        self.onehot_enc = OneHotEncoder()
        self.onehot_classes = self.onehot_enc.fit_transform(self.classes.reshape(-1, 1))
        self.n_classes = self.onehot_classes.shape[1]

        self.n_images = self.curr_imgs.shape[0]

        self.converter = Converter.load(assets.get_model_dir(model_name), include_encoders=include_encoders)
        self.content_codes = self.converter.content_encoder.predict(self.curr_imgs)
        self.class_codes = self.converter.class_encoder.predict(self.curr_imgs)
        class_adain_params = self.converter.class_modulation.predict(self.class_codes)
        self.class_adain_params = class_adain_params.reshape(class_adain_params.shape[0], -1)
        self.curr_imgs, self.classes, self.onehot_classes, self.content_codes, self.class_codes, self.class_adain_params = \
            shuffle(self.curr_imgs, self.classes, self.onehot_classes, self.content_codes, self.class_codes,
                    self.class_adain_params)

    def train_content_classifier(self, n_epochs):
        model = self.get_model(self.content_codes.shape[1])
        callbacks = [EarlyStopping('val_accuracy', patience=10)]
        hist = model.fit(self.content_codes, self.onehot_classes, epochs=n_epochs, validation_split=0.3,
                         callbacks=callbacks)
        wandb.log({'content_classifier_val_acc': hist.history['val_accuracy'][-1],
                   'content_classifier_n_epoch': hist.epoch[-1]})

    def train_class_classifier(self, n_epochs):
        print(f'Class code size: {self.class_codes.shape[1]}')
        model = self.get_model(self.class_codes.shape[1])
        callbacks = [EarlyStopping('val_accuracy', patience=10)]
        hist = model.fit(self.class_codes, self.onehot_classes, epochs=n_epochs, validation_split=0.3,
                         callbacks=callbacks)
        wandb.log({'class_classifier_val_acc': hist.history['val_accuracy'][-1],
                   'class_classifier_n_epoch': hist.epoch[-1]})

    def get_model(self, input_dim):
        model = Sequential()
        model.add(Dense(units=256, activation='relu', input_dim=input_dim))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.n_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

# In[6]:


# cc = LORDContentClassifier(model_name='mnist_model_64', data_name = 'minst_10_train')
# cc = LORDContentClassifier(model_name='smallnorb_model', data_name = 'smallnorb_test')
# cc = LORDContentClassifier(model_name='smallnorb_model', data_name = 'smallnorb_strict_class_test')
# cc = LORDContentClassifier(model_name='emnist_model_frist_stage', data_name = 'emnist_test')
# cc = LORDContentClassifier(model_name='smallnorb_model_fxd', data_name = 'smallnorb_strict_class_fxd_train')
# cc = LORDContentClassifier(model_name='smallnorb_model_fxd', data_name = 'smallnorb_strict_class_fxd_test')
# cc = LORDContentClassifier(model_name='smallnorb_no_adain_scale', data_name = 'smallnorb_strict_class_fxd_test')
#
# cc.train_content_classifier(10000)
# cc.train_class_classifier(10000)
