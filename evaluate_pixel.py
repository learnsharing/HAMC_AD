import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.utils import DataLoader, Test_DataLoader
from utils import *
import glob
import argparse
from model.liteFlownet import lite_flownet as lite_flow
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter


sys.stdout = Logger('./AUC.txt')
parser = argparse.ArgumentParser(description="Attention_VAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=0, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model', default='./auto_eva')

args = parser.parse_args()

torch.manual_seed(2020)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

test_folder = args.dataset_path+args.dataset_type+"/testing/frames"

# Loading dataset
test_dataset = Test_DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
test_size = len(test_dataset)
print(test_size)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)
loss_func_mse = nn.MSELoss(reduction='none')


flow_net = lite_flow.Network()
flow_net.load_state_dict(torch.load('model/liteFlownet/network-default.pytorch'))
flow_net.cuda().eval()
# model.cuda()

def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, vis_num):

    for t_idx in range(vis_num):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        test_pred = score_map_list[t_idx]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img[test_pred == 0] = 0
        cv2.imwrite('./new_result/' + args.dataset_type + '/pred_mask/' + str(t_idx) + '.jpg', test_pred*255)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0
    return image_resized

res = []
model_dir = os.listdir(args.model_dir)
for model_name in model_dir:
    model = torch.load(args.model_dir + '/' + model_name)
    print(model_name.split('.')[0])

    model.cuda()
    model.eval()

    labels = np.load('./data/frame_labels_' + args.dataset_type + '.npy')
    if args.dataset_type == 'shanghai':
        labels = np.expand_dims(labels, 0)


    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_list = {}

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name]['length'] + label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']


    gt_mask_list = []
    score_map_list = []
    test_imgs = []

    for k,(imgs) in enumerate(test_batch):
        if k == label_length - 4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']

        torch.cuda.synchronize()
        imgs, masks = Variable(imgs[0]).cuda(), Variable(imgs[1]).cuda()
        test_imgs.extend(imgs[:,12:15].cpu().detach().numpy())
        if  model_name.split('.')[0] == 'model_1':
            real_mask = masks[:, 4:5].cpu().squeeze(0).detach().numpy()
            real_mask = (real_mask.transpose(1, 2, 0).astype('uint8'))
            cv2.imwrite('./new_result/'+ args.dataset_type + '/real_mask/' + str(k) + '.jpg', real_mask * 255)

        mask = np.array(masks.cpu()[:, 4:5].squeeze())
        gt_mask_list.extend(masks[:, 4:5].cpu().detach().numpy())
        start = time.time()
        outputs = model.forward(imgs[:,0:3*4], False)
        torch.cuda.synchronize()
        end = time.time()
        res.append(end - start)

        diff_map = torch.sum(torch.abs(outputs - imgs[:,12:]).squeeze(), 0)
        # score_map = gaussian_filter(diff_map.cpu().detach().numpy(), sigma=5)
        score_map_list.append(diff_map.cpu().detach().numpy())
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))

    time_sum = 0
    for i in res:
        time_sum += i
    anomaly_score_total_list = []

    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        anomaly_score_total_list += anomaly_score_list(psnr_list[video_name])
    anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))
    print('Frame level AUC: ', accuracy*100, '%')

    flatten_gt_mask_list = np.concatenate(np.array(gt_mask_list)).ravel()
    flatten_score_map_list = np.concatenate(score_map_list).ravel()

    # calculate pixel level ROCAUC
    from sklearn.utils.multiclass import type_of_target

    for un in range(flatten_gt_mask_list.shape[0]):
        if 0 < flatten_gt_mask_list[un]<=0.5:
            flatten_gt_mask_list[un] = 0
        if 0.5 < flatten_gt_mask_list[un] <=1:
            flatten_gt_mask_list[un] = 1

    precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    # visualize localization result
    visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, vis_num=test_size)
    frame_fpr, frame_tpr, thresholds = get_fpr_tpr(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))
    anomaly_score = np.squeeze(anomaly_score_total_list)

    father_list = []
    for thre in range(thresholds.shape[0]):
        frame_label = []
        father_list.append(frame_label)
        for anoscore in range(anomaly_score.shape[0]):
            if anomaly_score[anoscore] > thresholds[thre]:
                father_list[thre].append(float(0))
            else:
                father_list[thre].append(float(1))

    pred_mask_path = './new_result/' + args.dataset_type + '/pred_mask/'
    real_mask_path = './new_result/' + args.dataset_type + '/real_mask/'

    total_labels = np.array(father_list)
    gt_labels = labels_list
    save_lab = []
    for m in range(test_size):
        pred_mask = np_load_frame(pred_mask_path + str(m) + '.jpg', 240, 360)# Ped2
        real_mask = np_load_frame(real_mask_path + str(m) + '.jpg', 240, 360)

        pred_mask = np.squeeze(pred_mask[:, :, 0:1])
        real_mask = np.squeeze(real_mask[:, :, 0:1])
        real_num = np.sum(real_mask == 1)
        if gt_labels[m] == 1 and real_num > 0:
            for i in range(pred_mask.shape[0]):
                for j in range(pred_mask.shape[1]):
                    if pred_mask[i][j] > 1:
                        pred_mask[i][j] = 1
            sum = pred_mask + real_mask
            sum_num = np.sum(sum == 2)
            if sum_num < real_num * 0.4:
                save_lab.append(m)

    save_lab = np.array(save_lab)
    for tl in range(total_labels.shape[0]):
        for sl in range(save_lab.shape[0]):
            total_labels[tl][save_lab[sl]] = 0

    P = np.sum(gt_labels == 1)
    N = np.sum(gt_labels == 0)
    tpr_total = []
    fpr_total = []
    for new_tl in range(total_labels.shape[0]):
        tp = np.sum((total_labels[new_tl] + gt_labels) == 2)
        fp = np.sum(total_labels[new_tl] == 1) - tp
        if tp == 0:
            tpr = 0
        else:
            tpr = tp / P
        tpr_total.append(tpr)

        if fp == 0:
            fpr = 0
        else:
            fpr = fp / N
        fpr_total.append(fpr)

    pixel_auc = auc(np.array(fpr_total), np.array(tpr_total))
    print('Pixel level AUC: %.10f' % (pixel_auc))
