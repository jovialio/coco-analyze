#!/usr/bin/env python3

import argparse
import os
import sys
import time
import glob
import json
import cv2

import tensorflow as tf
import tensorlayer as tl

sys.path.append('../../')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from openpose_plus.inference.common import measure, plot_humans, read_imgfile, CocoPart
from openpose_plus.inference.estimator import TfPoseEstimator
from openpose_plus.models import get_model

tf.logging.set_verbosity(tf.logging.INFO)
tl.logging.set_verbosity(tl.logging.INFO)

def write_json(img_name, humans, json_out):

    coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    
    image_h, image_w = cv2.imread(img_name).shape[:2]

    image_id = os.path.basename(img_name).replace('.jpg', '').replace('COCO_val2014_', '').replace('COCO_train2014_', '')
    image_id = int(image_id.lstrip('0'))
    
    for h in humans:
        
        new_keypoints = []
        
        for i in coco_ids:
        
            if i not in h.body_parts.keys():
                new_keypoints.append(0)
                new_keypoints.append(0)
                new_keypoints.append(0)
            else:            
                new_keypoints.append(int(h.body_parts[i].x * image_w + 0.5))
                new_keypoints.append(int(h.body_parts[i].y * image_h + 0.5))
                new_keypoints.append(2)
        
        json_out.append({"image_id": image_id,
                         "category_id" : 1,
                         "keypoints" : new_keypoints,
                         "score" : h.score})


def inference(base_model_name, path_to_npz, data_format, input_files, plot, writejson):
    model_func = get_model(base_model_name)
    height, width = (368, 432)
    e = measure(lambda: TfPoseEstimator(path_to_npz, model_func, target_size=(width, height), data_format=data_format),
                'create TfPoseEstimator')

    t0 = time.time()
    
    json_out = []
    
    for idx, img_name in enumerate(input_files):
                    
        image = measure(lambda: read_imgfile(img_name, width, height, data_format=data_format), 'read_imgfile')
        humans, heatMap, pafMap = measure(lambda: e.inference(image), 'e.inference')
        tl.logging.info('got %d humans from %s' % (len(humans), img_name))
        if humans:
            for h in humans:
                tl.logging.debug(h)
        if plot:
            if data_format == 'channels_first':
                image = image.transpose([1, 2, 0])
            plot_humans(image, heatMap, pafMap, humans, '%02d' % (idx + 1), os.path.basename(img_name))
        if writejson:
            write_json(img_name, humans, json_out)
    
    if writejson:
        with open('vis/json_detection.json', 'w') as f:
            json.dump(json_out, f, ensure_ascii=False)
            
    tot = time.time() - t0
    mean = tot / len(input_files)
    tl.logging.info('inference all took: %f, mean: %f, FPS: %f' % (tot, mean, 1.0 / mean))


def parse_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    parser.add_argument('--images', type=str, default='', help='comma separate list of image filenames', required=True)
    parser.add_argument('--base-model', type=str, default='vgg', help='vgg | vggtiny | mobilenet')
    parser.add_argument('--data-format', type=str, default='channels_last', help='channels_last | channels_first.')
    parser.add_argument('--plot', type=bool, default=False, help='draw the results')
    parser.add_argument('--repeat', type=int, default=1, help='repeat the images for n times for profiling.')
    parser.add_argument('--limit', type=int, default=100, help='max number of images.')
    parser.add_argument('--writejson', type=bool, default=False, help='write results to json')

    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.images.endswith(('.jpg', '.JPG', '.png', '.PNG')):
        image_files = ([f for f in args.images.split(',') if f] * args.repeat)[:args.limit]
        inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot, args.writejson)
        
    elif os.path.isdir(args.images):
        image_files = []
        for image in glob.glob(os.path.join(args.images, '*.png')):
            image_files.append(image)
        for image in glob.glob(os.path.join(args.images, '*.PNG')):
            image_files.append(image)
        for image in glob.glob(os.path.join(args.images, '*.jpg')):
            image_files.append(image)
        for image in glob.glob(os.path.join(args.images, '*.JPG')):
            image_files.append(image)
            
        inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot, args.writejson)
        
    elif os.path.isfile(args.images):
        image_files = []
    
        if args.images.endswith('.json'):
            dataDir  = os.path.dirname(os.path.dirname(args.images))#'/home/std/data/work/Dennis/openpose-plus/data/mscoco2017'
            dataType = os.path.basename(args.images).split('_')[-1].replace('.json', '')#'train2017'
            annType  = os.path.basename(args.images)[:os.path.basename(args.images).rfind('_')] #'person_keypoints'

            annFile  = '%s/annotations/%s_%s.json'%(dataDir, annType, dataType)

            print("{:10}[{}]".format('annFile:',annFile))

            with open(annFile) as f:
                gt_data = json.load(f)
            
            imgs_info = {i['id']:{'id':i['id'] ,
                                  'width':i['width'],
                                  'height':i['height']}
                                   for i in gt_data['images']}
            
            for annid in imgs_info.keys():
            
                if dataType.endswith('2017'):
                    filename = str(annid).zfill(12)+'.jpg'
                    image_files.append('{}/{}/{}'.format(dataDir, dataType, filename))
                elif dataType.endswith('2014'):
                    if dataType.startswith('val'):
                        filename = 'COCO_val2014_'+str(annid).zfill(12)+'.jpg'
                    elif dataType.startswith('train'):
                        filename = 'COCO_train2014_'+str(annid).zfill(12)+'.jpg'
                        
                    image_files.append('{}/{}/{}'.format(dataDir, dataType, filename))
            inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot, args.writejson)

        elif args.images.endswith('.txt'):

            with open(images, 'r') as textfile:
                for line in textfile:
                    image_files.append(line.strip())
            inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot, args.writejson)
        
        else:
            print('Wrong file format input. Only accepts .json or .txt file')

if __name__ == '__main__':
    measure(main)
