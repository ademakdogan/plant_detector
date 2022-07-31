#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""


import argparse
import pandas as pd
import os
import json

# image_name, page_width, page_height, x, y, width, height, labels
class DataConverter:
    
    def __init__(self, annotation_csv_path, is_train = True):
        
        self.ann = pd.read_csv(annotation_csv_path)
        self.is_train = is_train
        self.label_list = list(self.ann['labels'].unique())
        self.label2id = {}
        self.final_data = {}
    
    @staticmethod
    def get_final_path(sub_count, join_list):
    
        path = os.path.dirname(os.path.realpath(__file__))
        for i in range(sub_count):path = os.path.dirname(os.path.normpath(path))
        for i in range(len(join_list)):path = os.path.join(path, join_list[i])
        
        return path
    
    # 1-Categories
    def categories(self):
        
        cat = []
        for i in range(len(self.label_list)): 
            cat.append({"id" : i, "name" : str(self.label_list[i]), "supercategory": "N/A"})
            self.label2id[self.label_list[i]] = i
        self.final_data["categories"] = cat
        
        return
    
    # 2-Images
    def images(self):
        
        image_list = []
        images = list(self.ann['image_name'].unique())
        for i in range(len(images)):
        
            image_list.append({"id" : i,
                               "file_name" : images[i],
                               "width" : int(self.ann.loc[self.ann["image_name"] == images[i]].iloc[0]["page_width"]),
                               "height" : int(self.ann.loc[self.ann["image_name"] == images[i]].iloc[0]["page_height"]),
                               "data_captured" : "",
                               "licence" : 1,
                               "coco_url" : "",
                               "flickr_url" : ""})
        self.final_data["images"] = image_list
        
        return pd.DataFrame(image_list)

    # 3-Annotations
    def annotations(self, image_id_data):
        
        annotation_list = []
        for i in range(len(self.ann)):
            
            annotation_list.append({"id" : i,
                                    "image_id" : int(image_id_data.loc[image_id_data["file_name"] == self.ann["image_name"][i]].iloc[0]["id"]),
                                    "category_id" : self.label2id[self.ann["labels"][i]],
                                    "iscrowd" : 0,
                                    "area" : int(self.ann["width"][i]) * int(self.ann["height"][i]),
                                    "bbox" : [int(self.ann["x"][i]), int(self.ann["y"][i]), int(self.ann["width"][i]), int(self.ann["height"][i])]})
            
        self.final_data["annotations"] = annotation_list
        
        return
    
    def main(self):
        
        if self.is_train == True:
            print('...TRAIN...')
        else:
            print('...TEST...')
        print("Starting...")
        self.categories()
        print("1/3 The categories have been completed.")
        image_id_data = self.images()
        print("2/3 The images have been completed.")
        self.annotations(image_id_data)
        print("3/3 Annotations have been completed.")
        print("Done.")

        if self.is_train == True:
            data_path = DataConverter.get_final_path(1, ['data', 'json_files', 'custom_train.json'])
        else:
            data_path = DataConverter.get_final_path(1, ['data', 'json_files', 'custom_test.json'])
        with open(data_path, 'w') as fp:
                json.dump(self.final_data, fp)

        return self.final_data

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--annotation_csv_path", required=True, help="csv path")
    ap.add_argument("-i", "--is_train", required=True, help="true or false")

    args = vars(ap.parse_args())
    is_train = True
    if args['is_train'].lower() == 'false': is_train = False
    DataConverter(args['annotation_csv_path'], is_train).main()
