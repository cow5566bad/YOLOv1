import torch
import glob
import cv2
from torch.utils import data
from torchvision import transforms


class mydataset(data.Dataset):

    def __init__(self, image_dir, label_dir, image_size = 448, train = True):
        
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.train = train
        self.classes = ["plane", "ship", "storage-tank", "baseball-diamond",
                "tennis-court", "basketball-court", "ground-track-field", "harbor",
                "bridge", "small-vehicle", "large-vehicle", "helicopter",
                "roundabout", "soccer-ball-field", "swimming-pool", "container-crane"]
        self.boxes = [] # bboxes for all images
        # load images
        self.filenames = glob.glob(self.image_dir + '/*.jpg')
        self.filenames.sort()

        # load labels
        self.labelnames = glob.glob(self.label_dir + '/*.txt')
        self.labelnames.sort()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

        for label in self.labelnames:
            with open(label) as f:
                lines = f.readlines()
            box = [] # not only one box, but all boxes for an image
            for line in lines:
                splited = line.strip().split(" ")
                #splited: [xmin ymin xmax ymin xmax ymax xmin ymax cname difficulty]
                xmin, xmax = float(splited[0]), float(splited[2])
                ymin, ymax = float(splited[1]), float(splited[5])
                class_name = splited[8]
                box.append([xmin, ymin, xmax, ymax,
                        self.classes.index(class_name)])
            self.boxes.append(box)
        assert len(self.filenames) == len(self.labelnames)
        assert len(self.filenames) == len(self.boxes)

    def __len__(self):
        return  len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labelnames[index]
        img = cv2.imread(filename)
        img = cv2.resize(img, (self.image_size, self.image_size),
                interpolation=cv2.INTER_CUBIC)
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        rgb_img = torch.Tensor(rgb_img) / 255
        rgb_img = rgb_img.transpose(1, 2)
        rgb_img = rgb_img.transpose(0, 1)
        rgb_img = self.normalize(rgb_img)
        target = self.encode(self.boxes[index], index)
        return rgb_img, target

    def encode(self, boxes, cidx):
        '''
        boxes:
        box [xmin, ymin, xmax, ymax, index(class_name)] x number of boxes]
        in this image
        '''
        grid_num = 7
        target = torch.zeros(grid_num, grid_num, 26)
        cell_size = self.image_size / grid_num
        
        for box in boxes:
            # box is marked in 512*512, but image has been scaled down to 
            # 448 * 448, thereforem we need some scaling
            box = [b * (self.image_size/512) for b in box[:4]] + [box[4]]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            grid_x = (center_x // cell_size) * cell_size
            grid_y = (center_y // cell_size) * cell_size
            x = (center_x - grid_x) / cell_size
            y = (center_y - grid_y) / cell_size
            w = (box[2] - box[0]) / self.image_size
            h = (box[3] - box[1]) / self.image_size
            # i for y, j for x
            i, j = int(center_y // cell_size), int(center_x // cell_size)
            for idx, num in enumerate([x, y, w, h , 1]):
                target[i, j, idx] = num
            for idx, num in enumerate([x, y, w, h, 1]):
                target[i, j, 5+idx] = num
            target[i, j, 10 + box[4]] = 1
        return target

class test_dataset(data.Dataset):

    def __init__(self, image_dir, store_label_dir, image_size = 448, train = False):
        
        self.image_dir = image_dir
        self.store_label_dir = store_label_dir
        self.image_size = image_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        self.train = train
        
        # load images
        self.filenames = glob.glob(self.image_dir + '/*.jpg') 
        self.filenames.sort()
        
    def __len__(self):
        return  len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = cv2.imread(filename)
        img = cv2.resize(img, (self.image_size, self.image_size),
                interpolation=cv2.INTER_CUBIC)
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        rgb_img = torch.Tensor(rgb_img) / 255
        rgb_img = rgb_img.transpose(1, 2)
        rgb_img = rgb_img.transpose(0, 1)
        rgb_img = self.normalize(rgb_img)
        return rgb_img

def test():
    from torch.utils.data import DataLoader
    data = mydataset("hw2_train_val/train15000/images/",
            "hw2_train_val/train15000/labelTxt_hbb/")
    dataloader = DataLoader(data, batch_size = 1, shuffle=False)
    for i, pack in enumerate(dataloader):
        if i == 1: break
        img, target = pack
        img = img[i].numpy()
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        img = img.astype('int')
        r, g, b = cv2.split(img)
        img = cv2.merge([b, g, r])
        cv2.imwrite('test.jpg', img)
        

        for j in range(7):
            for k in range(7):
                print('loaded target:', target[i, j, k])
                xy = box_to_xy(j, k, target[i, j, k, :4])
                print(xy*(512/448))
def box_to_xy(i, j, box):
    grid_x = j * 64
    xmax = (448*box[2]+2*(64*box[0]+grid_x))/2
    xmin = xmax-448*box[2]
    
    grid_y = i * 64
    ymax = (448*box[3]+2*(64*box[1]+grid_y))/2
    ymin = ymax-448*box[3]
    return torch.Tensor([xmin, ymin,  xmax, ymax])

if __name__ == '__main__':
    test()
