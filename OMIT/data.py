import numpy as np
import onnx

import onnxruntime as ort
from tqdm import tqdm
from PIL import Image
import torch
from anti_face_recg_process_st_pth import score,get_model
from torch.utils.data import DataLoader
import os
import argparse
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from core.model import build_model
from core.checkpoint import CheckpointIO
from os.path import join as ospj
from core.face_reg_irnet import get_model
def transform( img):
    """
    输入图像预处理
    """
    img = img[..., ::-1].astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return img


class TorchDataset(Dataset):
    def __init__(self, filename):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_label_list = self.read_file(filename)
        self.len = len(self.image_label_list)

        self.transform_RGB = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        self.toTensor = transforms.ToTensor()

        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
    def __getitem__(self, i):
        index = i % self.len
        image_name, label = self.image_label_list[index]
        image_path = image_name
        img = Image.open(image_path)
        img=self.transform_RGB(img)
        return img, label
    def __len__(self):
        data_len = len(self.image_label_list)
        return data_len
    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip()
                image_label_list.append((content,content.split(".")[0]+"_IR.jpg"))
        return image_label_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=0,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=50000,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--mode', type=str,default=True,

                        help='This argument is used in solver')
    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default=r'C:\Users\deng\Desktop\val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/results',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints_psgan',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=2501)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=100000)
    network_identity = get_model("r50", fp16=False).to("cuda")
    network_identity.load_state_dict(torch.load("backbone.pth"))
    network_identity.eval()
    for param in network_identity.parameters():
        param.requires_grad = False
    args = parser.parse_args()
    _, nets = build_model(args)
    ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'), data_parallel=True, **nets)]
    for ckptio in ckptios:
        ckptio.load(args.resume_iter)
    for net in nets.keys():
        nets[net].eval()
        for param in nets[net].parameters():
            param.requires_grad = False
    transform_RGB = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    os.makedirs(args.result_dir, exist_ok=True)
    # xiacaiyang=torch.nn.functional.interpolate(input, size=112,  mode='bilinear')
    txt_path=r"F:\jishulin_data\paddle1\img_list2.txt"
    dataset=TorchDataset(txt_path)
    net_face = get_model('r50', fp16=False).to("cuda")
    net_face.load_state_dict(torch.load('backbone.pth'))
    net_face.eval()
    f=open(r"D:\wenjian\label.txt","w")
    dataloader=DataLoader(dataset, 64, shuffle=False, num_workers=8)
    for i in tqdm(dataloader):
        img,file_name=i
        img=img.to("cuda")
        y = torch.zeros(size=[img.shape[0]], dtype=torch.long).to("cuda")
        face_vector = face_vector=torch.randn(size=[128,64]).to("cuda")
        s_many = nets.mapping_network(face_vector, y)
        img_fake_IR = nets.generator(img, s_many, masks=None)
        scor = score(img_fake_IR, img, net_face)
        img_fake_IR=torch.nn.functional.interpolate(img_fake_IR, size=112,  mode='bilinear')#(img_fake_IR)
        src_np_hwc = np.transpose(img_fake_IR.data.cpu().numpy(), [0,2, 3, 1])
        src_np_img = np.array((np.clip(src_np_hwc, -1, 1) + 1) / 2 * 255, 'uint8')
        for l,img in enumerate(src_np_img):#, (img,file_name)
            score1,file_name_=scor[l],file_name[l]
            cv2.imwrite(file_name_, img[..., ::-1])
            f.write(file_name_ + "\t" + str(score1)[1:-1] + "\n")
