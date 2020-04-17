train = []
test = []

with open('train_test_split.txt', 'r')as f:
    for i in f.readlines():
        idx, flag = i.strip().split(' ')
        if int(flag):
            train.append(idx)
        else:
            test.append(idx)

print(len(train), len(test))


images = {}
with open('images.txt', 'r')as f:
    for i in f.readlines():
        idx, path = i.strip().split(' ')
        images[idx] = path

print(len(images))

labels = {}
with open('image_class_labels.txt', 'r')as f:
    for i in f.readlines():
        idx, label = i.strip().split(' ')
        # label in annotation is start with 1
        labels[idx] = int(label) - 1

print(len(labels))

bbox = {}
with open('bounding_boxes.txt', 'r')as f:
    for i in f.readlines():
        idx, x, y, w, h = i.strip().split(' ')
        bbox[idx] = [x, y, w, h]

print(len(bbox))


if __name__ == '__main__':
    # path, label
    with open('train.txt', 'w')as f:
        for idx in train:
            f.write('{},{}\n'.format(images[idx], labels[idx]))

    with open('test.txt', 'w')as f:
        for idx in test:
            f.write('{},{}\n'.format(images[idx], labels[idx]))


    # with open('train_bbox.txt', 'w')as f:
    #     for idx in train:
    #         x, y, w, h = bbox[idx]
    #         f.write('{},{},{},{},{},{}\n'.format(images[idx], labels[idx], x, y, w, h))
    #
    # with open('test_bbox.txt', 'w')as f:
    #     for idx in test:
    #         x, y, w, h = bbox[idx]
    #         f.write('{},{},{},{},{},{}\n'.format(images[idx], labels[idx], x, y, w, h))





