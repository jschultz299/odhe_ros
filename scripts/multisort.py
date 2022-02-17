# Test of multisort

import sys
from operator import attrgetter

def sort_detections(class_id, class_name, score, box, mask):

    # print_result(target)
    pass

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

class Detection:
    def __init__(self, id, name, x, y):
        self.id = id
        self.name = name
        self.x = x
        self.y = y
    def __repr__(self):
        return repr([self.id, self.name, self.x, self.y])


def main():

    detections_objects = [Detection(1, 'pretzel', 8, 0), Detection(2, 'pretzel', 3, 4), Detection(3, 'pretzel', 5, 2), Detection(4, 'pretzel', 3, 0), Detection(5, 'pretzel', 8, 4),]
    print("Original:      ", detections_objects)
    print("Sorted (x):    ", sorted(detections_objects, key=attrgetter('x')))
    print("Sorted (y):    ", sorted(detections_objects, key=attrgetter('y')))
    print("Sorted (x, y): ", sorted(detections_objects, key=attrgetter('x', 'y')))
    print("Sorted (y, x): ", sorted(detections_objects, key=attrgetter('y', 'x')))

    print("------------------------------------")

    detections_objects = [Detection(1, 'pretzel', 7, 4), Detection(2, 'pretzel', 9, 6), Detection(3, 'pretzel', 3, 1), Detection(4, 'pretzel', 11, 8), Detection(5, 'pretzel', 5, 3),]
    print("Original:      ", detections_objects)
    print("Sorted (x):    ", sorted(detections_objects, key=attrgetter('x')))
    print("Sorted (y):    ", sorted(detections_objects, key=attrgetter('y')))
    print("Sorted (x, y): ", sorted(detections_objects, key=attrgetter('x', 'y')))
    print("Sorted (y, x): ", sorted(detections_objects, key=attrgetter('y', 'x')))

    print("------------------------------------")

    detections_objects = [Detection(1, 'pretzel', 9, 2), Detection(2, 'pretzel', 11, 0), Detection(3, 'pretzel', 5, 6), Detection(4, 'pretzel', 3, 8), Detection(5, 'pretzel', 7, 4),]
    print("Original:      ", detections_objects)
    print("Sorted (x):    ", sorted(detections_objects, key=attrgetter('x')))
    print("Sorted (y):    ", sorted(detections_objects, key=attrgetter('y')))
    print("Sorted (x, y): ", sorted(detections_objects, key=attrgetter('x', 'y')))
    print("Sorted (y, x): ", sorted(detections_objects, key=attrgetter('y', 'x')))

    print("------------------------------------")

    detections_objects = [Detection(1, 'pretzel', 3, 3), Detection(2, 'pretzel', 1, 1), Detection(3, 'pretzel', 5, 1), Detection(4, 'pretzel', 3, 1), Detection(5, 'pretzel', 1, 3), Detection(6, 'pretzel', 5, 3),]
    print("Original:      ", detections_objects)
    print("Sorted (x):    ", sorted(detections_objects, key=attrgetter('x')))
    print("Sorted (y):    ", sorted(detections_objects, key=attrgetter('y')))
    print("Sorted (x, y): ", sorted(detections_objects, key=attrgetter('x', 'y')))
    print("Sorted (y, x): ", sorted(detections_objects, key=attrgetter('y', 'x')))


if __name__ == '__main__':
    sys.exit(main())
