'''
Descripttion: 
version: 
Author: BaoLu li
Date: 2021-08-26 14:36:21
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2022-02-17 00:19:26
'''

# -*- coding: utf-8 -*- 

import pdb

# object2 = [ box, class_id, class_name, score, mask ]

def sort_detections(class_id, class_name, score, box, mask):
    target = Out_transfer(class_id, class_name, score, box, mask)
    Sort_quick(target, 0, len(target)-1, y=True)

    #after sort the y, then start sorting the x:
    # arr_y = [(target[w][3][1]+target[w][3][3])/2 for w in range(len(target))]
    arr_y = [target[w][3][1] for w in range(len(target))]
    store = []

    for i in range(len(arr_y)):
        if arr_y.count(arr_y[i]) > 1:
            store.append([i, arr_y.count(arr_y[i])+1])

    if len(store) !=0:
        for each_group in store:
            Sort_quick(target, each_group[0], each_group[1], y=False)

    print_result(target)

    return target


def Out_transfer(class_id, class_name, score, box, mask):

    num = int(len(class_id))
    target = []
    target1 = []

    for i in range(num):

        target.append([class_id[i], class_name[i], score[i], box[i], mask[i]])

    return target




def partition(target, low, high, y=True): #low: the beginning index; high: the last index

    i = ( low-1 )
    arr = []
    # pdb.set_trace()
    
    if y:
        # arr = [(target[w][3][0]+target[w][3][2])/2 for w in range(len(target))] #box:[x1, y1, x2, y2]  value :(x1+x2)/2
        arr = [target[w][3][0]for w in range(len(target))]
    else:
        # arr = [(target[w][3][1]+target[w][3][3])/2 for w in range(len(target))] #box:[x1, y1, x2, y2]  value :(y1+y2)/2
        arr = [target[w][3][1] for w in range(len(target))]

    pivot = arr[high]

    for j in range(low , high): 
  
        if   arr[j] <= pivot: 
          
            i = i+1 
            target[i],target[j] = target[j],target[i] 
  
    target[i+1],target[high] = target[high],target[i+1] 

    return ( i+1 )


def Sort_quick(target, low, high, y):

    if low < high: 
        pi = partition(target,low,high, y) 
  
        Sort_quick(target, low, pi-1, y) 
        Sort_quick(target, pi+1, high, y)
        
def print_result(result):
    cls_id = list()
    cls_name = list()
    for i in range(len(result)):
        cls_id.append(result[i][0])
        cls_name.append(result[i][1])

    print("Original: ", class_id)
    print("Goal: ", goal)
    print("Sorted: ", cls_id)
    if goal == cls_id:
        success = True
    else:
        success = False
    print("Result: ", success)
    print("---------------------")


if __name__ == '__main__':

    # Example 1
    # class_id = [3, 3, 4, 3, 4, 3, 3]
    # class_name = ['pretzel', 'pretzel', 'gripper', 'pretzel', 'gripper', 'pretzel', 'pretzel']
    # score = [0.9998739957809448, 0.9998607635498047, 0.99965500831604, 0.9996339082717896, 0.9995941519737244, 0.9994165897369385, 0.9990149736404419]
    # box = [[15,10,15,10],[30,20,30,20],[20,10,20,10],[5,5,5,5],[1,5,5,1],[287,92,52,11],[3,5,3,5]]
    # mask = [[1.9998739957809448], [2.9998607635498047], [3.99965500831604], [4.9996339082717896], [5.9995941519737244], [6.9994165897369385], [7.9990149736404419]]
    # goal = ???

    # Example 2
    class_id = [1, 2, 3, 4, 5]
    class_name = ['pretzel', 'pretzel', 'gripper', 'pretzel', 'gripper']
    score = [0.9998739957809448, 0.9998607635498047, 0.99965500831604, 0.9996339082717896, 0.9995941519737244]
    box = [[8,0,9,1],[3,4,4,5],[5,2,6,3],[3,0,4,1],[8,4,9,5]]
    mask = [[1.9998739957809448], [2.9998607635498047], [3.99965500831604], [4.9996339082717896], [5.9995941519737244]]
    goal = [4,1,3,2,5]

    sort_detections(class_id, class_name, score, box, mask)

    # Example 3
    class_id = [1, 2, 3, 4, 5]
    class_name = ['pretzel', 'pretzel', 'gripper', 'pretzel', 'gripper']
    score = [0.9998739957809448, 0.9998607635498047, 0.99965500831604, 0.9996339082717896, 0.9995941519737244]
    box = [[7,4,8,5],[9,6,10,7],[3,1,4,2],[11,8,12,9],[5,3,6,4]]
    mask = [[1.9998739957809448], [2.9998607635498047], [3.99965500831604], [4.9996339082717896], [5.9995941519737244]]
    goal = [3, 5, 1, 2, 4]

    sort_detections(class_id, class_name, score, box, mask)

    # Example 4
    class_id = [1, 2, 3, 4, 5]
    class_name = ['pretzel', 'pretzel', 'gripper', 'pretzel', 'gripper']
    score = [0.9998739957809448, 0.9998607635498047, 0.99965500831604, 0.9996339082717896, 0.9995941519737244]
    box = [[9,2,10,3],[11,0,12,1],[5,6,6,7],[3,8,4,9],[7,4,8,5]]
    mask = [[1.9998739957809448], [2.9998607635498047], [3.99965500831604], [4.9996339082717896], [5.9995941519737244]]
    goal = [2, 1, 5, 3, 4]

    sort_detections(class_id, class_name, score, box, mask)

    # Example 5
    class_id = [1, 2, 3, 4, 5, 6]
    class_name = ['pretzel', 'pretzel', 'gripper', 'pretzel', 'gripper', 'pretzel']
    score = [0.9998739957809448, 0.9998607635498047, 0.99965500831604, 0.9996339082717896, 0.9995941519737244, 0.9998739957809448]
    box = [[3,3,4,4],[1,1,2,2],[5,1,6,2],[3,1,4,2],[1,3,2,4],[5,3,6,4]]
    mask = [[1.9998739957809448], [2.9998607635498047], [3.99965500831604], [4.9996339082717896], [5.9995941519737244],[1.9998739957809448]]
    goal = [2,4,3,5,1,6]

    result = sort_detections(class_id, class_name, score, box, mask)



