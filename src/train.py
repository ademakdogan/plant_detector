#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""

from coco_detection import CocoDetection
from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor
from detr import Detr
from pytorch_lightning import Trainer
import argparse
import os
import json
import torch
from datasets_helper import get_coco_api_from_dataset
from datasets_helper.coco_eval import CocoEvaluator
from tqdm import tqdm


class Train:
    
    def __init__(self, train_img_path, test_img_path):
        
        self.train_img_path = train_img_path
        self.test_img_path = test_img_path
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        config_path = Train.get_final_path(1, ['config.json'])
        with open(config_path, 'r') as f: config = json.load(f)
        self.train_batch_size = config['train_batch_size']
        self.test_batch_size = config['test_batch_size']
        self.gpus = config['gpus']
        self.lr = config['lr']
        self.lr_backbone = config['lr_backbone']
        self.weight_decay = config['weight_decay']
        self.max_steps = config['max_steps']
        self.gradient_clip_val = config['gradient_clip_val']
        
        '''
        from sklearn.model_selection import train_test_split
        ann = pd.read_csv(annotation_csv_path)
        if split_ratio == 0:
            DataConverter(ann).main()
        else:
            y = ann[['labels']]
            X = ann.drop('labels', axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio)
            X_train['labels'] = y_train['labels']
            X_test['labels'] = y_test['labels']
            X_train = X_train.reset_index(drop = True)
            X_test = X_test.reset_index(drop = True)
            DataConverter(X_train).main()
            DataConverter(X_test, is_train = False).main()
        '''

    @staticmethod
    def get_final_path(sub_count, join_list):
    
        path = os.path.dirname(os.path.realpath(__file__))
        for i in range(sub_count):path = os.path.dirname(os.path.normpath(path))
        for i in range(len(join_list)):path = os.path.join(path, join_list[i])
        
        return path
    
    @staticmethod
    def collate_fn(batch):
        
        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        pixel_values = [item[0] for item in batch]
        encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        
        return batch

    def get_json_files(self):
        
        train_path = Train.get_final_path(1, ['data', 'json_files', 'custom_train.json'])
        test_path = Train.get_final_path(1, ['data', 'json_files', 'custom_test.json'])
        model_path = Train.get_final_path(1, ['model','model.ckpt'])
        
        return train_path, test_path, model_path
    
    def create_dataset(self, train_path, test_path):

        #feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        train_dataset = CocoDetection(self.train_img_path, train_path, test_path, feature_extractor = self.feature_extractor)
        val_dataset = CocoDetection(self.test_img_path, train_path, test_path, feature_extractor = self.feature_extractor, train=False)
        
        return train_dataset, val_dataset
    
    def evaluation(self, val_dataset, val_dataloader, model):

        base_ds = get_coco_api_from_dataset(val_dataset)
        iou_types = ['bbox']
        coco_evaluator = CocoEvaluator(base_ds, iou_types) # initialize evaluator with ground truths

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        print("Running evaluation...")

        for idx, batch in enumerate(tqdm(val_dataloader)):
            # get the inputs
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

            # forward pass
            outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = self.feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
            res = {target['image_id'].item(): output for target, output in zip(labels, results)}
            coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    
    def train(self, train_dataset, val_dataset):

        train_dataloader = DataLoader(train_dataset, collate_fn=Train.collate_fn, batch_size = self.train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, collate_fn=Train.collate_fn, batch_size = self.test_batch_size)
        batch = next(iter(train_dataloader))
        cats = train_dataset.coco.cats
        id2label = {k: v['name'] for k,v in cats.items()}
        model = Detr(lr=self.lr, lr_backbone=self.lr_backbone, weight_decay=self.weight_decay, id2label = id2label, train_dataloader = train_dataloader, val_dataloader = val_dataloader)
        #PATH = '/Users/.../aa.ckpt'
        #model = model.load_from_checkpoint(PATH,lr=self.lr, lr_backbone=self.lr_backbone, weight_decay=self.weight_decay, id2label = id2label, train_dataloader = train_dataloader, val_dataloader = val_dataloader)
        trainer = Trainer(gpus = self.gpus, max_steps = self.max_steps, gradient_clip_val = self.gradient_clip_val)
        trainer.fit(model)

        #-----
        self.evaluation(val_dataset, val_dataloader, model)
        
        return model, trainer
        
    def main(self):
        
        
        train_path, test_path, model_path = self.get_json_files()
        train_dataset, test_dataset = self.create_dataset(train_path, test_path)
        _, trainer = self.train(train_dataset, test_dataset)
        trainer.save_checkpoint(model_path)
        
        return
    
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--train_img_path", required=True, help="images folder for train")
    ap.add_argument("-t", "--test_img_path", required=True, help="images folder for test")
    #ap.add_argument("-s", "--split_ratio", required=False, default = 0, help='split ratio', type = float)
    args = vars(ap.parse_args())
    Train(args['train_img_path'], args['test_img_path']).main()





        
      