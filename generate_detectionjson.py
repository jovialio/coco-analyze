#!/usr/bin/env python3

import argparse
import os
import sys
import time
import glob
import json
import cv2
from pprint import pprint
import numpy as np

import tensorflow as tf
import tensorlayer as tl

## COCO imports
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.cocoanalyze import COCOanalyze

sys.path.append('../../')

from openpose_plus.inference.common import measure, plot_humans, read_imgfile, CocoPart
from openpose_plus.inference.estimator import TfPoseEstimator
from openpose_plus.models import get_model
from pycocotools.coco import COCO

tf.logging.set_verbosity(tf.logging.INFO)
tl.logging.set_verbosity(tl.logging.INFO)

def round_int(val):
    return int(round(val))

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
                new_keypoints.append(round_int(h.body_parts[i].x * image_w))
                new_keypoints.append(round_int(h.body_parts[i].y * image_h))
                new_keypoints.append(2)

        json_out.append({"image_id": image_id,
                         "category_id" : 1,
                         "keypoints" : new_keypoints,
                         "score" : h.score})


def inference(base_model_name, path_to_npz, data_format, input_files, plot, writejson, outputdir, preExp13, is_resize = False):
    
    NumberOfIteration = path_to_npz[int(path_to_npz.rfind('pose'))+4:int(path_to_npz.rfind('.npz'))]

    model_func = get_model(base_model_name, is_resize)

    if is_resize:
        #Recommends : 368x432 or 368x656 or 736x1312
        width, height = (432,368)
    else:
        width, height = (None,None)
    
    e = measure(lambda: TfPoseEstimator(path_to_npz, model_func, target_size=(width, height), data_format=data_format),
                    'create TfPoseEstimator')
#    var_vgg19_stage1_kernel = []
#    var_vgg19_stage1_bias = []
#    var_stage2_kernel = []
#    var_stage2_bias = []
#
#    for variable in tf.trainable_variables():
#
#
#        # stage = 1 and vgg layer
#        if variable.name.find('vgg19') != -1 or variable.name.find('stage1') != -1:
#            
#            if variable.name.find('kernel') != -1:
#                var_vgg19_stage1_kernel.append(variable)
#            elif variable.name.find('bias') != -1:
#                var_vgg19_stage1_bias.append(variable)

#        # stage > 1
#        elif variable.name.find('stage2') != -1 or variable.name.find('stage3') != -1 or variable.name.find('stage4') != -1 or variable.name.find('stage5') != -1 or variable.name.find('stage6') != -1:
#            if variable.name.find('kernel') != -1:
#                var_stage2_kernel.append(variable)
#            elif variable.name.find('bias') != -1:
#                var_stage2_bias.append(variable)


#    print('var_vgg19_stage1_kernel')
#    pprint(var_vgg19_stage1_kernel)
#    print('var_vgg19_stage1_bias')
#    pprint(var_vgg19_stage1_bias)
#    print('var_stage2_kernel')
#    pprint(var_stage2_kernel)
#    print('var_stage2_bias')
#    pprint(var_stage2_bias)

    #print([n.name for n in tf.get_default_graph().as_graph_def().node])
    t0 = time.time()

    json_out = []

    for idx, img_name in enumerate(input_files):

        image, orig_image = measure(lambda: read_imgfile(img_name, width, height, preExp13=preExp13, data_format=data_format), 'read_imgfile')
        
        if orig_image is None:
            sys.exit(-1)
                                   
        humans, heatMap, pafMap = measure(lambda: e.inference(image), 'e.inference')

        # for logging purposes        
        tl.logging.info('got %d humans from %s' % (len(humans), img_name))
        if humans:
            for h in humans:
                tl.logging.debug(h)
        if plot:
            if data_format == 'channels_first':
                image = image.transpose([1, 2, 0])
                orig_image = orig_image.transpose([1, 2, 0])
            plot_humans(orig_image, heatMap, pafMap, humans, '%02d' % (idx + 1), os.path.basename(img_name), outputdir)
        if writejson:
            write_json(img_name, humans, json_out)

    if writejson:
        with open(outputdir+'/json_detection_'+NumberOfIteration+'.json', 'w') as f:
            json.dump(json_out, f, ensure_ascii=False)
            
    tot = time.time() - t0
    mean = tot / len(input_files)
    tl.logging.info('inference all took: %f, mean: %f, FPS: %f' % (tot, mean, 1.0 / mean))

    return outputdir+'/json_detection_'+NumberOfIteration+'.json'


def parse_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    parser.add_argument('--images', type=str, default='', help='comma separate list of image filenames', required=True)
    parser.add_argument('--base-model', type=str, default='vgg', help='vgg19 | vgg | vggtiny | mobilenet')
    parser.add_argument('--data-format', type=str, default='channels_last', help='channels_last | channels_first.')
    parser.add_argument('--plot', type=bool, default=False, help='draw the results')
    parser.add_argument('--repeat', type=int, default=1, help='repeat the images for n times for profiling.')
    parser.add_argument('--limit', type=int, default=100, help='max number of images.')
    parser.add_argument('--writejson', type=bool, default=False, help='write results to json')
    parser.add_argument('--outputdir', type=str, default='vis', help='write results to out dir')
    parser.add_argument('--gpu', type=str, default='0', help='write results to out dir')
    parser.add_argument('--preExp13', type=str, default='None', help='Type True or False')
    parser.add_argument('--iter', type=int, default=0, help='iteration to start inference')

    return parser.parse_args()


def main():
    args = parse_args()

    annFile  = args.images

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # if os.path.isdir(args.outputdir):
    # 	for f in glob.glob(os.path.join(args.outputdir,'*')):
    # 		os.remove(f)
    # else:
    # 	os.mkdir(args.outputdir)
    
    if args.images.endswith(('.jpg', '.JPG', '.png', '.PNG')):
        image_files = ([f for f in args.images.split(',') if f] * args.repeat)[:args.limit]
        inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot, args.writejson, args.outputdir.strip('/'), args.preExp13)
        
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
            
        inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot, args.writejson, args.outputdir.strip('/'), args.preExp13)
        
    elif os.path.isfile(args.images):
        image_files = []
    
        if args.images.endswith('.json'):
            print('Running Coco dataset.')
            dataDir  = os.path.dirname(os.path.dirname(args.images))#'/home/std/data/work/Dennis/openpose-plus/data/mscoco2017'
            dataType = os.path.basename(args.images).split('_')[-1].replace('.json', '')#'train2017'
            annType  = os.path.basename(args.images)[:os.path.basename(args.images).rfind('_')] #'person_keypoints'

            annFile  = '%s/annotations/%s_%s.json'%(dataDir, annType, dataType)

            print("{:10}[{}]".format('annFile:',annFile))

            cocoGt = COCO(annFile)
            catIds = cocoGt.getCatIds(catNms=['person'])
            keys = cocoGt.getImgIds(catIds=catIds)
                        
            for annid in keys:
            
                img_meta = cocoGt.loadImgs(annid)[0]
            
                if dataType.endswith('2017'):
                    #filename = str(annid).zfill(12)+'.jpg'
                    #image_files.append('{}/{}/{}'.format(dataDir, dataType, filename))
                            
                    image_files.append('{}/{}/{}'.format(dataDir, dataType, img_meta['file_name']))
                elif dataType.endswith('2014'):
                    if dataType.startswith('val'):
                        filename = img_meta['file_name']#'COCO_val2014_'+str(annid).zfill(12)+'.jpg'
                    elif dataType.startswith('train'):
                        filename = img_meta['file_name']#'COCO_train2014_'+str(annid).zfill(12)+'.jpg'
                        
                    image_files.append('{}/{}/{}'.format(dataDir, dataType, filename))
            
            if os.path.isdir(args.path_to_npz):
                list_of_npz = sorted(glob.glob(args.path_to_npz+'*.npz'), key=os.path.getmtime)
                for npz_file in list_of_npz:  

                    iteration = int(npz_file[int(npz_file.rfind('pose'))+4:int(npz_file.rfind('.npz'))])
                    if iteration >= args.iter and iteration%2000 == 0:                        

                        resFile = inference(args.base_model, npz_file, args.data_format, image_files, args.plot, args.writejson, args.outputdir.strip('/'),args.preExp13)

                        ## load ground truth annotations
                        coco_gt = COCO(annFile)
                        catIds = coco_gt.getCatIds(catNms=['person'])
                        keys = coco_gt.getImgIds(catIds=catIds)

                        ## initialize COCO detections api
                        coco_dt   = coco_gt.loadRes(resFile)

                        ## initialize COCO eval api
                        coco_analyze = COCOanalyze(coco_gt, coco_dt, 'keypoints')
                        coco_analyze.evaluate(verbose=True, makeplots=True, savedir=resFile[:int(resFile.rfind('json_detection'))], team_name=str(iteration))
                        
            else:
                inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot, args.writejson, args.outputdir.strip('/'), args.preExp13)

        elif args.images.endswith('.txt'):

            with open(images, 'r') as textfile:
                for line in textfile:
                    image_files.append(line.strip())
            inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot, args.writejson, args.outputdir.strip('/'), args.preExp13)
        
        else:
            print('Wrong file format input. Only accepts .json or .txt file')

if __name__ == '__main__':
    measure(main)
