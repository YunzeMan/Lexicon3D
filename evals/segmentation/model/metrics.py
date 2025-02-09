import torch
import numpy as np


#NYU_CATEGORY_NAMES = ['wall', 'floor', 'cabinet', 'bed', 'chair',
#                      'sofa', 'table', 'door', 'window', 'bookshelf',
#                      'picture', 'counter', 'blinds', 'desk', 'shelves',
#                      'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
#                      'clothes', 'ceiling', 'books', 'refridgerator', 'television',
#                      'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
#                      'person', 'night stand', 'toilet', 'sink', 'lamp',
#                      'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
# NYU 40

NYU_CATEGORY_NAMES = ['wall', 'floor', 'cabinet', 'bed', 'chair',
                      'sofa', 'table', 'door', 'window', 'bookshelf',
                      'picture', 'counter', 'desk', 'curtain', 'refridgerator',
                      'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

class SemsegMeter(object):
    def __init__(self, database='NYUD', ignore_idx=255):
        ''' "marco" way in ATRC evaluation code.
        '''
        if database == 'PASCALContext':
            n_classes = 20
            cat_names = VOC_CATEGORY_NAMES
            has_bg = True
             
        elif database == 'NYUD':
            n_classes = 20
            cat_names = NYU_CATEGORY_NAMES
            has_bg = False

        else:
            raise NotImplementedError
        
        self.n_classes = n_classes + int(has_bg)
        self.cat_names = cat_names
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

        self.correct_total = 0
        self.point_total = 0

        self.ignore_idx = ignore_idx

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.squeeze()
        gt = gt.squeeze()
        valid = (gt != self.ignore_idx)
        self.point_total += torch.sum(valid).item()
        self.correct_total += torch.sum((pred == gt) & valid).item()
        for i_part in range(0, self.n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

        self.correct_total = 0
        self.point_total = 0
            
    def get_score(self, verbose=True):
        jac = [0] * self.n_classes
        acc_every_class = [0] * self.n_classes
        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)
            acc_every_class[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['mIoU'] = np.mean(jac)
        eval_result['mAcc'] = np.mean(acc_every_class)
        eval_result['Acc'] = self.correct_total / self.point_total 

        if verbose:
            print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(100 * eval_result['mIoU']))
            class_IoU = jac #eval_result['jaccards_all_categs']
            for i in range(len(class_IoU)):
                spaces = ''
                for j in range(0, 20 - len(self.cat_names[i])):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))
        print("Acc: ", eval_result['Acc'])
        print("mAcc: ", eval_result['mAcc'])
        return eval_result
