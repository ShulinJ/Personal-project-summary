import numpy as np
import onnx

import onnxruntime as ort
from tqdm import tqdm
from PIL import Image
import torch
from anti_face_recg_process_st_pth import score,get_model
from sklearn import preprocessing
import os
import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
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
    parser.add_argument('--resume_iter', type=int, default=30000,
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
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints_rgb',
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
    input_img_path = r"C:\Users\deng\Desktop\input\rgb"

    out_path=r"C:\Users\deng\Desktop\input\ir"
    dirs=os.walk(input_img_path)
    img_size = (112, 112)
    i=0
    sum=0
    net_face = get_model('r50', fp16=False).to("cuda")
    net_face.load_state_dict(torch.load('backbone.pth'))
    net_face.eval()

    f=open(r"F:\jishulin_data\paddle1\score.txt","w")
    # f2 = open(r"F:\jishulin_data\wrong.txt", "w")
    a=0
    all=0
    b=0
    for roots,_,files in tqdm(dirs):
        # try:
            list_img=[]
            batch_img =None
            file_img_list=[]
            for file in files:
                    if "nm" in file or "IR" in file:
                        os.remove(os.path.join(roots,file))
                    else:
                        img_path = os.path.join(roots, file)
                        img = Image.open(img_path)
                        img=transform_RGB(img).unsqueeze(0)
                        if batch_img==None:
                            batch_img=img
                        else:
                            batch_img=torch.cat((batch_img,img), 0)
                        file_img_list.append(file)
            if batch_img is not None:
                a=0
                for i in range(0,batch_img.shape[0],80):
                    batch_80_img=batch_img[i:min(batch_img.shape[0],i+80)].to("cuda")
                    y = torch.zeros(size=[min(batch_img.shape[0],i+80)-i], dtype=torch.long).to("cuda")
                    # s_many = nets.style_encoder(batch_80_img, y)
                    face_vector=torch.randn(size=[min(batch_img.shape[0],i+80)-i,16]).to("cuda")
                    # face_vector=network_identity(batch_80_img)
                    s_many = nets.mapping_network(face_vector, y)

                    masks = nets.fan.get_heatmap(batch_80_img) if args.w_hpf > 0 else None

                    img_fake_IR=nets.generator(batch_80_img,s_many,masks=masks)
                    for img_fake,img_real in zip(img_fake_IR,batch_80_img):
                        scor=score(img_fake.unsqueeze(0),img_real.unsqueeze(0),net_face)
                        src_np_hwc=np.transpose(img_fake.data.cpu().numpy(),[1,2,0])
                        src_np_hwc=cv2.resize(src_np_hwc,(112,112))
                        src_np_img = np.array((np.clip(src_np_hwc, -1, 1) + 1) / 2 * 255, 'uint8')
                        cv2.imwrite(os.path.join(out_path,file_img_list[a].split(".")[0]+"_IR.jpg"), src_np_img[..., ::-1])
                        all=all+scor
                        a+=1
                        b+=1
            print(all/b)