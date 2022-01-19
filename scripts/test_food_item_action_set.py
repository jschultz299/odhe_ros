#!/usr/bin/env python
# Food item action set Matrix

import pickle

food_items = list()

item = dict()
item['id'] = 0
item['name'] = 'Baby_Carrot'
item['class'] = 'carrot'
item['form'] = 'baby'
item['group'] = 'single'
item['compliance'] = 'low'
item['roughness'] = 'low'
item['friction'] = 'low'
item['action'] = 'grasp'
food_items.append(item)

item = dict()
item['id'] = 1
item['name'] = 'Pretzel_Rod'
item['class'] = 'pretzel'
item['form'] = 'rod'
item['group'] = 'single'
item['compliance'] = 'low'
item['roughness'] = 'medium'
item['friction'] = 'high'
item['action'] = 'grasp'
food_items.append(item)

item = dict()
item['id'] = 2
item['name'] = 'Celery_Stick'
item['class'] = 'celery'
item['form'] = 'stick'
item['group'] = 'single'
item['compliance'] = 'low'
item['roughness'] = 'medium'
item['friction'] = 'medium'
item['action'] = 'grasp'
food_items.append(item)

with open("/home/labuser/ros_ws/src/odhe_ros/scripts/food_items.pkl","wb") as file:
        pickle.dump(food_items, file)    # Although intuitively should use the max, the min tends to yield the best results
file.close()