import numpy as np
import os
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import cv2
import shutil
import argparse

def compute_scores(y_test, y_pred):
    aps = average_precision_score(y_test, y_pred)
    r_p, r_r, r_t = precision_recall_curve(y_test, y_pred)
    r_f1 = 2*r_r*r_p/(r_r + r_p)
    r_f1[np.isnan(r_f1)] = 0
    # print(r_p, r_r, r_t)
    print("Average Precision Score:", aps)
    print('Best F1 threshold: ', r_t[np.argmax(r_f1)])
    print('Best F1-Score: ', np.max(r_f1))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_test, [m >= thresh for m in y_pred]))

    accuracies = np.array(accuracy_scores)
    max_accuracy = accuracies.max() 
    max_accuracy_threshold =  thresholds[accuracies.argmax()]
    print("Max Accuracy Threshold:", max_accuracy_threshold)
    print("Max Accuracy:", max_accuracy)
    

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('path_to_data', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

# args = parser.parse_args()

base_path = './'
removed_pid_list = base_path + '/annotations/removed_pids.txt'
annotations_path = base_path + '/annotations/'
data_path = base_path + '/data'
removed_pids = []
with open(removed_pid_list, 'r') as f:
    for line in f.readlines():
        removed_pids.append(line.strip())
    f.close()

flag = 0
for i in os.listdir(annotations_path):
    if 'gt_annotations_6_' in i:
        if flag == 0:
            df = pd.read_csv(os.path.join(annotations_path, i))
            flag = 1
        else:
            df = pd.concat([df, pd.read_csv(os.path.join(annotations_path, i))])

# map annotator id to annotations
pid_dict = {}
for p, pid in enumerate(df['pid'].tolist()):
    if str(pid) == 'nan':
        continue
    else:
        pid_dict[df['uniqueid'].tolist()[p]] = pid

#count total number of annotators
count_pids = 0
for pid in pid_dict:
    if pid_dict[pid] in removed_pids:
        count_pids += 1


# mapping annotations to image names
img_dict = {}
for i, img_name in enumerate(df['image_name'].tolist()):
    if df['uniqueid'].tolist()[i] not in pid_dict:
        continue
    if pid_dict[df['uniqueid'].tolist()[i]] in removed_pids:
        continue
    if str(img_name) == 'nan':
        continue
    else:
        if img_name in img_dict:
            img_dict[img_name].append(df['annotations'].tolist()[i])
        else:
            img_dict[img_name] = [df['annotations'].tolist()[i]]
    


train_imgs = np.load(os.path.join(base_path, 'train_images.npy'))
removal_gt_scores = []
retention_gt_scores = []
plant_gt_scores = []
plant_scores = []
perturb_scores = []
removal_scores = []
retention_scores = []
distance_gt_score = []
distance_score = []
removal_distance_score = []
vehicle_type = []
im_per_vehicle = []
everything_important = []
plant_vehicle_type = []

count = 0
userid = '8_15_qualitative' #output_directory
phase = 'test' #or test
all_imgs = []

for img in img_dict:
    if phase == 'train':
        if img not in train_imgs:
            continue
    else:
        if img in train_imgs:
            continue

    # remove images with less than 5 annotations
    if len(img_dict[img]) < 5:
        continue

    # remove the practice round images
    if 'trial' in img:
        continue

    count += 1
    
    dir_name = img.split("_")[2]
    img_number = img.split("_")[-1]

    
    data_file = img.split(".")[0].split("_")[-1] + '.npy'
    dir_name = img.split("_")[2]
    data = np.load(os.path.join(data_path, dir_name, "data", data_file), allow_pickle=True)
    plant_data = np.load(os.path.join(data_path, dir_name, "data", "plant_data_" + data_file), allow_pickle=True)
    img_number2 = img.split("_")[-1].split('.')[0]
    
    # removing an extra image that incorrectly got in the dataset
    if img_number2 == "2":
        continue
    
    # mapping between original simulation frames and the recorded frames
    recording_mapping = {2: 17, 9: 15, 14: 2, 24:21, 31:18, 19:20}
    
    all_imgs.append(img)
    bbs = img_dict[img]
    for bbox in bbs:
        if bbox == '[]':
            continue
        b = bbox[2:-2].split('], [')
        for j in range(len(b)):
            bb = [int(k.split('.')[0]) for k in b[j].split(',')]
    try:
        new_data = np.load(os.path.join(data_path, dir_name, "recording_data", "data", str(int(img_number2) - recording_mapping[int(dir_name)]) + '.npy'), allow_pickle=True)
    except:
        continue

    vehicle_positions = data[-1]
    
    vehicle_dict = {}
    for bbs in img_dict[img]:
    # read annotated vehicles
        if bbs == '[]':
            continue
        b = bbs[2:-2].split('], [')
        
        for j in range(len(b)):
            # print([k for k in bbs[j].split(',')])
            bb = [int(k.split('.')[0]) for k in b[j].split(',')]
            # print(bb)
            for vehicle in vehicle_positions:
                #xywh
                if vehicle_positions[vehicle][0] >= bb[0] and vehicle_positions[vehicle][0] <= bb[0] + bb[2] and vehicle_positions[vehicle][1] >= bb[1] and vehicle_positions[vehicle][1] <= bb[1] + bb[3]:            
                    if vehicle not in vehicle_dict:
                        vehicle_dict[vehicle] = 1
                    else:
                        vehicle_dict[vehicle] += 1

    perturb_important_vehicles_raw = new_data[-2]
    perturb_important_vehicles_bbs = new_data[-3]
    project_important_vehicles_raw = new_data[-4]
    vehicle_data = {}

    for j in new_data[-1]:
        vehicle_data[j] = new_data[-1][j][-1]

    #diamondback, gazelle, crossbike, walker
    perturb_important_vehicles = {}
    project_important_vehicles = {}
    for x in project_important_vehicles_raw:        
        # removing a perturbation type we tried but didnt improve results
        if x[4] == 3:
            continue

        
        if x[0] in project_important_vehicles:
            project_important_vehicles[x[0]] = min(project_important_vehicles[x[0]], x[-1])
        else:
            project_important_vehicles[x[0]] = x[-1]
    vehicle_positions = data[-1]
    for j, vehicle_info in enumerate(perturb_important_vehicles_raw):
        if vehicle_info[4] == 3:
            continue
        vehicle_bb = perturb_important_vehicles_bbs[j]
        xs = np.array([vehicle_bb[b][0] for b in range(4)])
        ys = np.array([vehicle_bb[b][1] for b in range(4)])
        for vehicle in vehicle_positions:
            if vehicle_positions[vehicle][0] <= np.max(xs) and vehicle_positions[vehicle][0] >= np.min(xs) and vehicle_positions[vehicle][1] <= np.max(ys) and vehicle_positions[vehicle][1] >= np.min(ys):
                if vehicle in perturb_important_vehicles:
                    perturb_important_vehicles[vehicle] = min(vehicle_info[-1], perturb_important_vehicles[vehicle])
                else:
                    perturb_important_vehicles[vehicle] = vehicle_info[-1]
    
    thresh1 = int(len(img_dict[img])*0.6)
    thresh2 = int(len(img_dict[img])*0.2)
    # thresh1 = 0
    # thresh2 = -1
    removal_important_vehicles = data[2]
    ego_location = (640, 365)

    for vehicle in removal_important_vehicles:
        if vehicle[0] in vehicle_dict:
            if vehicle_dict[vehicle[0]] >= thresh1:     
                if vehicle[0] in vehicle_positions:
                    if (vehicle_positions[vehicle[0]][0] >= 0 and vehicle_positions[vehicle[0]][0] <= 720) and (vehicle_positions[vehicle[0]][1] >= 0 and vehicle_positions[vehicle[0]][1] <= 1280):
                        everything_important.append(1)
                    else:
                        everything_important.append(0)
                else:
                    everything_important.append(0) 
                if vehicle[0] in vehicle_positions[vehicle[0]]:
                    im_per_vehicle.append((img, vehicle_positions[vehicle[0]]))
                else:
                    im_per_vehicle.append((img, "None"))

                if vehicle[0] in vehicle_data:
                    if 'pedestrian' in vehicle_data[vehicle[0]] or 'diamondback' in vehicle_data[vehicle[0]] or 'gazelle' in vehicle_data[vehicle[0]] or 'crossbike' in vehicle_data[vehicle[0]]:
                        vehicle_type.append('p')
                        if vehicle[0] in vehicle_positions:
                            if (vehicle_positions[vehicle[0]][0] >= 0 and vehicle_positions[vehicle[0]][0] <= 720) and (vehicle_positions[vehicle[0]][1] >= 0 and vehicle_positions[vehicle[0]][1] <= 1280):
                                removal_distance_score.append(-1*np.linalg.norm(np.array(list(vehicle_positions[vehicle[0]])) - np.array(list(ego_location))))
                            else:
                                removal_distance_score.append(-10000)
                        else:
                            removal_distance_score.append(-10000)
                    else:                      
                        vehicle_type.append('v')
                        removal_distance_score.append(-10000)
                else:
                    vehicle_type.append('v')
                    removal_distance_score.append(-10000)

                removal_gt_scores.append(1)
                removal_scores.append(vehicle[1])
                if vehicle[0] in perturb_important_vehicles and vehicle[0] in project_important_vehicles:                    
                    perturb_scores.append(-1*min(perturb_important_vehicles[vehicle[0]], project_important_vehicles[vehicle[0]]))
                elif vehicle[0] in perturb_important_vehicles:
                    perturb_scores.append(-1* perturb_important_vehicles[vehicle[0]])
                elif vehicle[0] in project_important_vehicles:
                    perturb_scores.append(-1*project_important_vehicles[vehicle[0]])
                else:
                    perturb_scores.append(-20)
            elif vehicle_dict[vehicle[0]] <= thresh2:
                if vehicle[0] in vehicle_positions:
                    if (vehicle_positions[vehicle[0]][0] >= 0 and vehicle_positions[vehicle[0]][0] <= 720) and (vehicle_positions[vehicle[0]][1] >= 0 and vehicle_positions[vehicle[0]][1] <= 1280):
                        everything_important.append(1)
                    else:
                        everything_important.append(0)
                else:
                    everything_important.append(0) 
                
                if vehicle[0] in vehicle_positions:
                    im_per_vehicle.append((img, vehicle_positions[vehicle[0]]))
                else:
                    im_per_vehicle.append((img, "None"))
                if vehicle[0] in vehicle_data:
                    if 'pedestrian' in vehicle_data[vehicle[0]] or 'diamondback' in vehicle_data[vehicle[0]] or 'gazelle' in vehicle_data[vehicle[0]] or 'crossbike' in vehicle_data[vehicle[0]]:
                        vehicle_type.append('p')
                        if vehicle[0] in vehicle_positions:
                            if (vehicle_positions[vehicle[0]][0] >= 0 and vehicle_positions[vehicle[0]][0] <= 720) and (vehicle_positions[vehicle[0]][1] >= 0 and vehicle_positions[vehicle[0]][1] <= 1280):
                                removal_distance_score.append(-1*np.linalg.norm(np.array(list(vehicle_positions[vehicle[0]])) - np.array(list(ego_location))))
                            else:
                                removal_distance_score.append(-10000)
                        else:
                            removal_distance_score.append(-10000)
                    else:
                        vehicle_type.append('v')
                        removal_distance_score.append(-10000)
                else:
                    vehicle_type.append('v')
                    removal_distance_score.append(-10000)
                removal_gt_scores.append(0)
                removal_scores.append(vehicle[1])
                if vehicle[0] in perturb_important_vehicles and vehicle[0] in project_important_vehicles:                    
                    perturb_scores.append(-1*min(perturb_important_vehicles[vehicle[0]], project_important_vehicles[vehicle[0]]))
                elif vehicle[0] in perturb_important_vehicles:
                    perturb_scores.append(-1* perturb_important_vehicles[vehicle[0]])
                elif vehicle[0] in project_important_vehicles:
                    perturb_scores.append(-1*project_important_vehicles[vehicle[0]])
                else:
                    perturb_scores.append(-20)
            
        else:
            if vehicle[0] in vehicle_positions:
                if (vehicle_positions[vehicle[0]][0] >= 0 and vehicle_positions[vehicle[0]][0] <= 720) and (vehicle_positions[vehicle[0]][1] >= 0 and vehicle_positions[vehicle[0]][1] <= 1280):
                    everything_important.append(1)
                else:
                    everything_important.append(0)
            if vehicle[0] in vehicle_positions:
                    im_per_vehicle.append((img, vehicle_positions[vehicle[0]]))
            else:
                im_per_vehicle.append((img, "None"))
            removal_gt_scores.append(0)


            removal_scores.append(vehicle[1])
            if vehicle[0] in perturb_important_vehicles and vehicle[0] in project_important_vehicles:                    
                perturb_scores.append(-1*min(perturb_important_vehicles[vehicle[0]], project_important_vehicles[vehicle[0]]))
            elif vehicle[0] in perturb_important_vehicles:
                    perturb_scores.append(-1* perturb_important_vehicles[vehicle[0]])
            elif vehicle[0] in project_important_vehicles:
                perturb_scores.append(-1*project_important_vehicles[vehicle[0]])
            else:
                perturb_scores.append(-20)
                # perturb_scores.append(0)
            
            if vehicle[0] in vehicle_data:
                if 'pedestrian' in vehicle_data[vehicle[0]] or 'diamondback' in vehicle_data[vehicle[0]] or 'gazelle' in vehicle_data[vehicle[0]] or 'crossbike' in vehicle_data[vehicle[0]]:
                    vehicle_type.append('p')
                    
                    if vehicle[0] in vehicle_positions:
                        if (vehicle_positions[vehicle[0]][0] >= 0 and vehicle_positions[vehicle[0]][0] <= 720) and (vehicle_positions[vehicle[0]][1] >= 0 and vehicle_positions[vehicle[0]][1] <= 1280):
                            removal_distance_score.append(-1*np.linalg.norm(np.array(list(vehicle_positions[vehicle[0]])) - np.array(list(ego_location))))
                        else:
                            removal_distance_score.append(-10000)
                    else:
                        removal_distance_score.append(-10000)
                else:
                    vehicle_type.append('v')
                    removal_distance_score.append(-10000)
            else:
                vehicle_type.append('v')
                removal_distance_score.append(-10000)
        

    # Retention based important vehicles
    retention_important_vehicles = data[3]
    for vehicle in retention_important_vehicles:
        if vehicle[0] in vehicle_dict:
            if vehicle_dict[vehicle[0]] >= thresh1:
                retention_gt_scores.append(1)
                retention_scores.append(vehicle[1])
            elif vehicle_dict[vehicle[0]] <= thresh2:
                retention_gt_scores.append(0)
                retention_scores.append(vehicle[1])
        else:
            retention_gt_scores.append(0)
            retention_scores.append(vehicle[1])


    #vehicle location
    ego_location = (640, 365)
    for vehicle in vehicle_positions:
        # print(vehicle_positions[vehicle])
        if int(vehicle) in vehicle_dict:

            if vehicle_dict[int(vehicle)] >= thresh1:
                distance_gt_score.append(1)
                # print(vehicle_positions[vehicle])
                distance_score.append(-1*np.linalg.norm(np.array(list(vehicle_positions[vehicle])) - np.array(list(ego_location))))
            elif vehicle_dict[int(vehicle)] <= thresh2:
                distance_gt_score.append(0)
                distance_score.append(-1*np.linalg.norm(np.array(list(vehicle_positions[vehicle])) - np.array(list(ego_location))))
        else:
            distance_gt_score.append(0)
            distance_score.append(-1*np.linalg.norm(np.array(list(vehicle_positions[vehicle])) - np.array(list(ego_location))))

    for v, vehicle in enumerate(plant_data[0][0]):
        if int(vehicle) in vehicle_dict:
            if vehicle_dict[int(vehicle)] >= thresh1:
                if vehicle in vehicle_data:
                    if 'pedestrian' in vehicle_data[vehicle] or 'diamondback' in vehicle_data[vehicle] or 'gazelle' in vehicle_data[vehicle] or 'crossbike' in vehicle_data[vehicle]:
                        plant_vehicle_type.append('p')
                    else:
                        plant_vehicle_type.append('v')
                else:
                    plant_vehicle_type.append('v')
                plant_gt_scores.append(1)
                plant_scores.append(plant_data[0][1][v])
            elif vehicle_dict[int(vehicle)] <= thresh2:
                if vehicle in vehicle_data:
                    if 'pedestrian' in vehicle_data[vehicle] or 'diamondback' in vehicle_data[vehicle] or 'gazelle' in vehicle_data[vehicle] or 'crossbike' in vehicle_data[vehicle]:
                        plant_vehicle_type.append('p')
                    else:
                        plant_vehicle_type.append('v')
                else:
                    plant_vehicle_type.append('v')
                plant_gt_scores.append(0)
                plant_scores.append(plant_data[0][1][v])
        else:
            if vehicle in vehicle_data:
                if 'pedestrian' in vehicle_data[vehicle] or 'diamondback' in vehicle_data[vehicle] or 'gazelle' in vehicle_data[vehicle] or 'crossbike' in vehicle_data[vehicle]:
                    plant_vehicle_type.append('p')
                else:
                    plant_vehicle_type.append('v')
            else:
                plant_vehicle_type.append('v')
            plant_gt_scores.append(0)
            plant_scores.append(plant_data[0][1][v])
    vehicle_positions_og = data[-1]
    ps = []
    for v, vehicle in enumerate(plant_data[0][0]):
        ps.append(plant_data[0][1][v])
    plant_img = cv2.imread('/home/harpadmin/vehicle_importance_human_annotation/psiturk-harp-template/static/stimuli_images/' + img)
    for j, vehicle in enumerate(plant_data[0][0]):
        if int(vehicle) in vehicle_positions_og:
            vehicle_positions = vehicle_positions_og[int(vehicle)]
            plant_img = cv2.rectangle(plant_img, tuple([vehicle_positions[0]-10, vehicle_positions[1]-10]), tuple([vehicle_positions[0]+10, vehicle_positions[1]+10]), (255, 0, 0)*int(min(ps[j], 1)), 2)

# removal_scores = np.minimum(removal_scores, 20)
# retention_scores = np.minimum(retention_scores, 20)

print("Results for both vehicles and images")

print("Distance")
a = (np.array(distance_score) - np.min(distance_score))/(np.max(distance_score) - np.min(distance_score))
compute_scores(np.array(distance_gt_score), np.array(a))

print("Plant")
compute_scores(np.array(plant_gt_scores), np.array(plant_scores))
print("\n")

# b -> pedestrian (inverse + removal + velocity), vehicle (removal + velocity)
a = np.maximum(np.array(removal_scores)/np.max(removal_scores), (np.array(perturb_scores) - np.min(perturb_scores))/(np.max(perturb_scores) - np.min(perturb_scores)))
b = np.maximum((np.array(removal_distance_score) - np.min(removal_distance_score))/((0.65/0.35)*(np.max(removal_distance_score) - np.min(removal_distance_score))) ,np.maximum(np.array(removal_scores)/np.max(removal_scores), (np.array(perturb_scores) - np.min(perturb_scores))/(np.max(perturb_scores) - np.min(perturb_scores))))
c = (np.array(distance_score) - np.min(distance_score))/(np.max(distance_score) - np.min(distance_score))
for i in range(len(a)):
    if a[i] != b[i]:
        a[i] = c[i]/(0.65/0.35)
print("Our Score")
compute_scores(np.array(removal_gt_scores), a)
print("\n")

print("Everyting Important")
compute_scores(np.array(removal_gt_scores), np.array(everything_important))
print("\n")


vehicle_removal_score = []
vehicle_retention_score = []
vehicle_perturb_score = []
vehicle_distance_score = []
vehicle_plant_score = []
vehicle_everything_important = []
vehicle_removal_distance_score = []

gt_vehicle_removal_score = []
gt_vehicle_retention_score = []
gt_vehicle_perturb_score = []
gt_vehicle_distance_score = []
gt_vehicle_plant_score = []


for i in range(len(vehicle_type)):
    if vehicle_type[i] == 'v':
        vehicle_removal_score.append(removal_scores[i])
        vehicle_retention_score.append(retention_scores[i])
        vehicle_perturb_score.append(perturb_scores[i])
        vehicle_distance_score.append(distance_score[i])
        vehicle_everything_important.append(everything_important[i])
        # vehicle_removal_distance_score.append(removal_distance_score[i])

        gt_vehicle_removal_score.append(removal_gt_scores[i])
        gt_vehicle_retention_score.append(retention_gt_scores[i])
        gt_vehicle_distance_score.append(distance_gt_score[i])

for i in range(len(plant_vehicle_type)):
    if plant_vehicle_type[i] == 'v':
        vehicle_plant_score.append(plant_scores[i])
        gt_vehicle_plant_score.append(plant_gt_scores[i])

print("Vehicle Only Scores\n")

print("Everything Important")        
compute_scores(np.array(gt_vehicle_removal_score), np.array(vehicle_everything_important))
print("\n")

print("Distance")
a = (np.array(vehicle_distance_score) - np.min(vehicle_distance_score))/(np.max(vehicle_distance_score) - np.min(vehicle_distance_score))
compute_scores(np.array(gt_vehicle_distance_score), np.array(a))
print("\n")


print("Plant")
# print(gt_vehicle_plant_score)
# print(vehicle_plant_score)
compute_scores(np.array(gt_vehicle_plant_score), np.array(vehicle_plant_score))
print("\n")


print("Our")
a = np.maximum(np.array(vehicle_removal_score)/np.max(vehicle_removal_score), (np.array(vehicle_perturb_score) - np.min(vehicle_perturb_score))/(np.max(vehicle_perturb_score) - np.min(vehicle_perturb_score)))
compute_scores(np.array(gt_vehicle_removal_score), np.maximum(np.array(vehicle_removal_score)/np.max(vehicle_removal_score), (np.array(vehicle_perturb_score) - np.min(vehicle_perturb_score))/(np.max(vehicle_perturb_score) - np.min(vehicle_perturb_score))))
print("\n")