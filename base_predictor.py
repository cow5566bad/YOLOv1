import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Yolov1_vgg16bn
from dataset import mydataset
from yoloLoss import yoloLoss

class Predictor():
    def __init__(self,
                batch_size = 15,
                max_epoches = 60,
                valid = None,
                device = None,
                learning_rate = 1e-3):
        self.batch_size = batch_size
        self.max_epoches = max_epoches
        self.valid = valid
        self.device = device
        self.learning_rate = learning_rate
        self.model = Yolov1_vgg16bn(pretrained=True)
        self.loss = yoloLoss(7, 2, 5, 0.5)
        self.best_valid_loss = 100
        self.image_size = 448
        # self.loss = torch.nn.MSELoss()

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() \
                                                else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.learning_rate,
                                        momentum=0.9,
                                        weight_decay=5e-4)
        self.epoch = 0
        logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    def save(self, path):
        torch.save({
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }, path)

    def load(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    def fit_dataset(self, data):
        self.model.train()
        while self.epoch < self.max_epoches:

            print('training %i' % self.epoch)
            dataloader = DataLoader(data, batch_size = self.batch_size, shuffle=True)
            self._run_epoch(dataloader, True)

            if self.valid is not None:
                print('validing %i' % self.epoch)
                validloader = DataLoader(self.valid, batch_size
                        = self.batch_size, shuffle=False)
                self._run_epoch(validloader, False)
            self.epoch += 1

    def _run_epoch(self, loader, training):
        loss = 0
        trange = tqdm(enumerate(loader), total = len(loader), desc='training')
        for i, batch in trange:
            if i >= len(loader): break

            if training:
                output, batch_loss = self._run_iter(batch, training)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output, batch_loss = self._run_iter(batch, training)

            loss += batch_loss.item()
            trange.set_postfix(loss=loss/(i+1))

        loss_in_epoch = loss/len(loader)
        if not training:
            if loss_in_epoch < self.best_valid_loss:
                self.best_valid_loss = loss_in_epoch
                logging.info('save best model at epoch %i :' % self.epoch)
                logging.info('best valid loss : %f' % self.best_valid_loss)
                self.save('yolo.pkl')

    def _run_iter(self, batch, training):
        with torch.no_grad():
            img, ground = batch
            img, ground = img.to(self.device), ground.to(self.device)

        pred = self.model.forward(img)
        loss = self.loss(pred, ground)
        return pred, loss

    def predict_dataset(self, test):
        self.model.eval()
        loader = DataLoader(
                test,
                batch_size = self.batch_size,
                shuffle = False)
        logging.info('load trained model...')
        self.load('yolo.pkl')
        logging.info('start predicting...')

        trange = tqdm(enumerate(loader), total = len(loader), desc='testing')
        output = []
        for i, batch in trange:
            if i >= len(loader): break
            with torch.no_grad():
                batch = batch.to(self.device)
                pred = self.model.forward(batch)
                for img_pred in pred:
                    output.append(img_pred)
        #output = [dataLenx7x7x26] tensor
        output = torch.stack(output, dim = 0)
        return self.decode(output)


    def decode(self, output):
        '''
        input: dataLen x 7 x 7 x 26 tensor
        return: dataLen x boxes
            boxes: [#box x [xmin, ymin, xmax, ymax, c, index]]
        '''

        grid_num = 7
        cell_size = self.image_size / grid_num
        threshold = 0.1

        output_decode= []

        trange = tqdm(range(output.shape[0]), total = output.shape[0], desc='decoding')
        for i in trange:
            # img_pred: 7 x 7 x 26
            img_pred = output[i, :, :, :]
            # turn img_pred into 98 x , 
            # 98 x [xmin, xmax, ymin, ymax, Pr(C_i|Object))]
            boxes = torch.zeros(2, 7, 7, 6)
            for i in range(grid_num):
                for j in range(grid_num):
                    for b in range(2):
                        PrC_max, PrC_max_index = torch.max(img_pred[i, j, 10:], 0)
                        PrC_max_index = PrC_max_index.float()
                        if b == 0:
                            score = img_pred[i, j, 4] * PrC_max
                        else:
                            score = img_pred[i, j, 9] * PrC_max
                        #if score1 < 0.1: score = 0.
                        box_xy = self.box_to_xy(i, j, img_pred[i, j, b*5:b*5+4])
                        boxes[b, i, j, :4] = box_xy
                        boxes[b, i, j, 4] = score
                        boxes[b, i, j, 5] = PrC_max_index
            boxes = boxes.view(-1, 6)
            keep = boxes[:, 4] > threshold
            boxes = boxes[keep]
            keep = self.nms(boxes)
            output_decode.append(boxes[keep])
        return output_decode

    def nms(self, boxes, threshold = 0.5):
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        areas = (x2-x1) * (y2-y1)
        scores = boxes[:, 4]
        _,order = scores.sort(0,descending=True)
        keep = []
        while order.numel() > 0:

            if order.dim() == 0:
                order = torch.Tensor([order])
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.numel() == 0:
                 break
            order = order[ids+1]
        return torch.LongTensor(keep)

    def box_to_xy(self, i, j, box):
        # box[x, y, w, h, c] Tensor
        grid_x = j * 64
        xmax = (448 * box[2] + 2 * (64 * box[0] + grid_x)) / 2
        xmin = xmax - 448 * box[2]

        grid_y = i * 64
        ymax = (448 * box[3] + 2 * (64 * box[1] + grid_y)) / 2
        ymin = ymax - 448 * box[3]
        return torch.Tensor([xmin, ymin, xmax, ymax])


def iou(box1, box2):
    
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    print(x1, y1, x2, y2)
    x5 = max(x1, x3);
    y5 = max(y1, y3);
    x6 = min(x2, x4);
    y6 = min(y2, y4);
    
    inter = (x6-x5) * (y6-y5)
    area1 = (x2-x1) * (y2-y1)
    area2 = (x4-x3) * (y4-y3)
    iou = inter / (area1 + area2 - inter)
    return iou

def test():
    # test of iou
    bbox1 = torch.Tensor([50, 50, 60, 60])
    bbox2 = torch.Tensor([55, 55, 65, 65])
    print(iou(bbox1, bbox2))

    #test of nms
    boxes = torch.Tensor([[50, 50, 60, 60, 0.8], [51, 51, 61, 61, 0.7], [49, 49, 59, 59, 0.6], [55,
        55, 65, 65, 0.75]])
    pd = Predictor()
    keep = pd.nms(boxes, 0.5)
    print(boxes[keep])

if __name__ == "__main__":
    test()
