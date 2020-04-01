import json
from PIL import Image
import numpy as np
import time
import os
import argparse

def argument_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--root', type=str, required=True,
                            help="root path to data directory")
    parser.add_argument('--category',  
                        type=int,  required=True, 
                        help="category of apparel")
    return parser                  

parser = argument_parser()
args = parser.parse_args()
                

dataset = {
    "annotations": [],
        }

print(args.root, args.category)

#CATEGORY LIST

#  1: short_sleeved_shirt
#  2: long_sleeved_shirt
#  3: short_sleeved_outwear
#  4: long_sleeved_outwear
#  5: vest
#  6: sling
#  7: shorts 
#  8: trousers  
#  9: skirt
#  10: short_sleeved_dress
#  11: long_sleeved_dress
#  12: vest_dress
#  13: sling_dress

st = time.time()
#num of images 
num_images = 170000
sub_index = 0 # the index of ground truth instance
for num in range(1,num_images+1):

    json_name = args.root + '/annos/' + str(num).zfill(6)+'.json'
    image_name = args.root + '/image/' + str(num).zfill(6)+'.jpg'
    image_file_name = str(num).zfill(6)+'.jpg'
    
    try:
        imag = Image.open(image_name)
        with open(json_name, 'r') as f:
            temp = json.loads(f.read())
    except:
        num = -1

    if num%10000==0:
        print("num", num)


    if (num>=0):
        
        width, height = imag.size
        pair_id = temp['pair_id']

        for i in temp:
            if i == 'source' or i=='pair_id':
                continue
            else:
                # points = np.zeros(14 * 3)
                # sub_index = sub_index + 1
                box = temp[i]['bounding_box']
                w = box[2]-box[0]
                h = box[3]-box[1]
                x_1 = box[0]
                y_1 = box[1]
                bbox=[x_1,y_1,w,h]
                cat = temp[i]['category_id']
                style = temp[i]['style']
                seg = temp[i]['segmentation']
                landmarks = temp[i]['landmarks']

                # points_x = landmarks[0::3]
                # points_y = landmarks[1::3]
                # points_v = landmarks[2::3]
                # points_x = np.array(points_x)
                # points_y = np.array(points_y)
                # points_v = np.array(points_v)

                # if cat == 1:
                #     for n in range(0, 25):
                #         points[3 * n] = points_x[n]
                #         points[3 * n + 1] = points_y[n]
                #         points[3 * n + 2] = points_v[n]
                # elif cat ==2:
                #     for n in range(25, 58):
                #         points[3 * n] = points_x[n - 25]
                #         points[3 * n + 1] = points_y[n - 25]
                #         points[3 * n + 2] = points_v[n - 25]
                # elif cat ==3:
                #     for n in range(58, 89):
                #         points[3 * n] = points_x[n - 58]
                #         points[3 * n + 1] = points_y[n - 58]
                #         points[3 * n + 2] = points_v[n - 58]
                # elif cat == 4:
                #     for n in range(89, 128):
                #         points[3 * n] = points_x[n - 89]
                #         points[3 * n + 1] = points_y[n - 89]
                #         points[3 * n + 2] = points_v[n - 89]
                # elif cat == 5:
                #     for n in range(128, 143):
                #         points[3 * n] = points_x[n - 128]
                #         points[3 * n + 1] = points_y[n - 128]
                #         points[3 * n + 2] = points_v[n - 128]
                # elif cat == 6:
                #     for n in range(143, 158):
                #         points[3 * n] = points_x[n - 143]
                #         points[3 * n + 1] = points_y[n - 143]
                #         points[3 * n + 2] = points_v[n - 143]
                # elif cat == 7:
                #     for n in range(158, 168):
                #         points[3 * n] = points_x[n - 158]
                #         points[3 * n + 1] = points_y[n - 158]
                #         points[3 * n + 2] = points_v[n - 158]
                # elif cat == 8:
                #     for n in range(25, 39): #168-182 14 points
                #         points[3 * n] = points_x[n - 25]
                #         points[3 * n + 1] = points_y[n - 25]
                #         points[3 * n + 2] = points_v[n - 25]
                # elif cat == 9:
                #     for n in range(182, 190):
                #         points[3 * n] = points_x[n - 182]
                #         points[3 * n + 1] = points_y[n - 182]
                #         points[3 * n + 2] = points_v[n - 182]
                # elif cat == 10:
                #     for n in range(190, 219):
                #         points[3 * n] = points_x[n - 190]
                #         points[3 * n + 1] = points_y[n - 190]
                #         points[3 * n + 2] = points_v[n - 190]
                # elif cat == 11:
                #     for n in range(219, 256):
                #         points[3 * n] = points_x[n - 219]
                #         points[3 * n + 1] = points_y[n - 219]
                #         points[3 * n + 2] = points_v[n - 219]
                # elif cat == 12:
                #     for n in range(256, 275):
                #         points[3 * n] = points_x[n - 256]
                #         points[3 * n + 1] = points_y[n - 256]
                #         points[3 * n + 2] = points_v[n - 256]
                # elif cat == 13:
                #     for n in range(275, 294):
                #         points[3 * n] = points_x[n - 275]
                #         points[3 * n + 1] = points_y[n - 275]
                #         points[3 * n + 2] = points_v[n - 275]
                # num_points = len(np.where(points_v > 0)[0])

            if cat in [args.category]:
                        
                dataset['annotations'].append({
                    'file_name': image_file_name,
                    'id': num,
                    'width': width,
                    'height': height,
                    'area': w*h,
                    'bbox': bbox,
                    'category_id': cat,
                    'image_id': num,
                    'iscrowd': 0,
                    'style': style, 
                    'keypoints':landmarks,
                    'segmentation': seg,
                    })

    
json_name = args.root + '/deepfashion2_datafile_'+ str(args.category) + '.json'
with open(json_name, 'w') as f:
    json.dump(dataset, f)

print("time taken", time.time() - st)