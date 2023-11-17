import random
import torchvision.transforms as transforms
from PIL import Image
import torch
from thop import profile
import cv2
import numpy as np
from torch.utils.data import sampler
import collections
from thop import profile,clever_format
import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.special import logsumexp
def recognition(config, img, model, converter, device,imgH,img_W):
    # imgH = 64
    h, w = img.shape
    # img_W = 220
    img = cv2.resize(img, (img_W, imgH))
    img = np.reshape(img, (imgH, img_W, 1))
    img = img.astype(np.float32)
    img = (img / 255. - config["test_option"]["mean"]) / config["test_option"]["std"]
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)
    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    # a, b = profile(model, img[None])
    # flops, params = clever_format([a, b], "%.3f")
    # print(flops, params)
    preds = model(img)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = torch.autograd.Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred

def DICT_create(cfg):
    with open(cfg["CHAR_FILE"], 'r',encoding="utf-8") as file:
        DICT = {char.replace("\n",""): num + 1 for num, char in enumerate(file.readlines())}
    NUM_CLASSES = len(DICT)
    ALPHABETS = ""
    for i in DICT.keys():
        ALPHABETS += i
    return DICT,ALPHABETS,NUM_CLASSES

'''
Could not load symbol cublasGetSmCountTarget from
Could_not_load_symbol_cublasGetSmCountTarget_from
'''



def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label
class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail )
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples
class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.toTensor = transforms.ToTensor()
    def __call__(self, batch):
        #################长文本
        # images, labels = zip(*batch)
        # imgH = self.imgH
        # imgW = self.imgW
        # images_out = []
        # labels_out = []
        # images_4he1_out=[]
        # labels_4he1_out=[]
        # for image, label in zip(images, labels):
        #         w, h = image.size
        #         img_W = int(imgH / h * w)
        #         if img_W > imgW or img_W < 20:
        #             continue
        #         image = np.array(image.resize((img_W, imgH), Image.BICUBIC))
        #         images_out.append(image)
        #         labels_out.append(label)
        # for i in range(0,len(images_out),5):
        #         images_4he1 = Image.fromarray(np.concatenate(images_out[i:min(i+5,len(images_out))], 1))
        #         # plt.imshow(images_4he1)
        #         # plt.show()
        #         lable="".join(j+" " for j in labels_out[i:min(i+5,len(images_out))])[:-1]
        #         # print(lable)
        #         if images_4he1.size[0]>2000:
        #             continue
        #             # images_4he1=images_4he1.resize((2000, 32), Image.BICUBIC)
        #         new_image = Image.new('L', (2000, 32), (0))
        #         new_image.paste(images_4he1, ((0, 0)))
        #         new_image = self.toTensor(new_image)
        #         new_image.sub_(0.5).div_(0.5)
        #         images_4he1_out.append(new_image)
        #         labels_4he1_out.append(lable)
        # images_out = torch.cat([t.unsqueeze(0) for t in images_4he1_out], 0)


########################不定长
        # images, labels = zip(*batch)
        # imgH = self.imgH
        # imgW = self.imgW
        # images_out=[]
        # labels_out=[]
        # if not self.keep_ratio:
        #     for image,label in zip(images,labels):
        #         new_image = Image.new('L', (imgW,imgH), (0))
        #         w, h = image.size
        #         img_W=int(imgH/h*w)
        #         if img_W >imgW or img_W<20:
        #             continue
        #         image = image.resize((img_W,imgH), Image.BICUBIC)
        #         new_image.paste(image, ((0, 0)))
        #         new_image = self.toTensor(new_image)
        #         new_image.sub_(0.5).div_(0.5)
        #         images_out.append(new_image)
        #         labels_out.append(label)
        #     images_out = torch.cat([t.unsqueeze(0) for t in images_out], 0)
        # else:
        #     for image,label in zip(images,labels):
        #         w, h = image.size
        #         img_W = int(imgH / h * w)
        #         image = image.resize((img_W, imgH), Image.BICUBIC)
        #         image = self.toTensor(image)
        #         image.sub_(0.5).div_(0.5)
        #         images_out.append(image)
        #         labels_out.append(label)
#################################原始
        images, labels_out = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images_out = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images_out, labels_out
class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, dict,alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        self.dict = dict
        self.alphabet = alphabet


    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            for char in text:
                if char not in self.dict:
                    print("{}这个字符提供的字典里没有，现将其修改.".format(char))
                    text=text.replace(char,"-")
        if isinstance(text, str):

            text = [
                        self.dict[char.lower() if self._ignore_case else char]
                        for char in text
                ]

            length = [len(text)]


        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                            char_list.append(self.alphabet[t[i] - 1])

                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
def loadData(v, data):
    v.resize_(data.size()).copy_(data)

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def val(device,data_loader,model, converter=None,criterion=None):
    print('Start val')
    model.eval()
    n_correct = 0
    all_numpy = 0
    val_epoch=min(10,len(data_loader))
    with torch.no_grad():
        for i, (cpu_images, cpu_texts) in enumerate(data_loader):
            for cpu_image,cpu_text in zip(cpu_images,cpu_texts):
                cpu_image=cpu_image.unsqueeze(0)
                batch_size = cpu_image.size(0)
                image=cpu_image.to(device)
                predss = model(image)
                preds_size = Variable(torch.IntTensor([predss.size(0)] * batch_size))
                _, preds = predss.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
                all_numpy += 1

                if sim_preds.lower() == cpu_text.lower():
                    n_correct += 1
                if i==val_epoch-1:
                    print('%-20s, gt: %-20s' % ( sim_preds.lower(), cpu_text.lower()))
            if i >val_epoch:
                break
    accuracy = n_correct / all_numpy
    print(' accuray: %f' % ( accuracy))
    return accuracy
def val_rnn(device,data_loader,model, converter=None,criterion=None):
    print('Start val')
    model.rnn.eval()
    n_correct = 0
    all_numpy = 0
    val_epoch=min(10,len(data_loader))
    with torch.no_grad():
        for i, (cpu_images, cpu_texts) in enumerate(data_loader):
            for cpu_image,cpu_text in zip(cpu_images,cpu_texts):
                cpu_image=cpu_image.unsqueeze(0)
                batch_size = cpu_image.size(0)
                image=cpu_image.to(device)
                predss = model(image)
                preds_size = Variable(torch.IntTensor([predss.size(0)] * batch_size))
                _, preds = predss.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
                all_numpy += 1
                if sim_preds.lower()== cpu_text.lower():
                    n_correct += 1
                if i==1:
                    print('%-20s, gt: %-20s' % ( sim_preds.lower(), cpu_text.lower()))
            if i > val_epoch:
                break
    accuracy = n_correct / all_numpy
    print(' accuray: %f' % ( accuracy))
    return accuracy
NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01
def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]
    return new_labels
def greedy_decode(emission_log_prob, blank=0, **kwargs):
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels
def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))
    length, class_count = emission_log_prob.shape
    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                # log(p1 * p2) = log_p1 + log_p2
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))
        # sorted by accumulated_log_prob
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
    # sum up beams to produce labels
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix))
        # log(p1 + p2) = logsumexp([log_p1, log_p2])
        total_accu_log_prob[labels] = \
            logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])
    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels


def prefix_beam_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs['beam_size']
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [(tuple(), (0, NINF))]  # (prefix, (blank_log_prob, non_blank_log_prob))
    # initial of beams: (empty_str, (log(1.0), log(0.0)))

    for t in range(length):
        new_beams_dict = collections.defaultdict(lambda: (NINF, NINF))  # log(0.0) = NINF

        for prefix, (lp_b, lp_nb) in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue

                end_t = prefix[-1] if prefix else None

                # if new_prefix == prefix
                new_lp_b, new_lp_nb = new_beams_dict[prefix]

                if c == blank:
                    new_beams_dict[prefix] = (
                        logsumexp([new_lp_b, lp_b + log_prob, lp_nb + log_prob]),
                        new_lp_nb
                    )
                    continue
                if c == end_t:
                    new_beams_dict[prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_nb + log_prob])
                    )

                # if new_prefix == prefix + (c,)
                new_prefix = prefix + (c,)
                new_lp_b, new_lp_nb = new_beams_dict[new_prefix]

                if c != end_t:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob, lp_nb + log_prob])
                    )
                else:
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob])
                    )

        # sorted by log(blank_prob + non_blank_prob)
        beams = sorted(new_beams_dict.items(), key=lambda x: logsumexp(x[1]), reverse=True)
        beams = beams[:beam_size]

    labels = list(beams[0][0])
    return labels

def ctc_decode(log_probs, DICT=None, blank=0, method='beam_search', beam_size=5):
    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    label2char = {}
    for key, val in DICT.items():
        label2char[val] = key
    decoders = {
        'greedy': greedy_decode,
        'beam_search': beam_search_decode,
        'prefix_beam_search': prefix_beam_decode,
    }
    decoder = decoders[method]

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if label2char:
            decoded = "".join(label2char[l] for l in decoded)
        decoded_list.append(decoded)
    return decoded_list