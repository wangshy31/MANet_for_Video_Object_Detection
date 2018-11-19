import numpy as np
import os
import cv2
import random
from PIL import Image
from bbox.bbox_transform import clip_boxes
from bbox.bbox_transform import bbox_transform
from bbox.bbox_transform import bbox_pred
import xml.etree.ElementTree as ET
import imageio
import cv2



# TODO: This two functions should be merged with individual data loader
def get_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

def get_pair_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]

        eq_flag = 0 # 0 for unequal, 1 for equal
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roi_rec.has_key('pattern'):
            ref_id = min(max(roi_rec['frame_seg_id'] + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET+1), 0),roi_rec['frame_seg_len']-1)
            ref_image = roi_rec['pattern'] % ref_id
            assert os.path.exists(ref_image), '%s does not exist'.format(ref_image)
            ref_im = cv2.imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if ref_id == roi_rec['frame_seg_id']:
                eq_flag = 1
        else:
            ref_im = im.copy()
            eq_flag = 1

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_eq_flags, processed_roidb

def check_movements(ims, bef_ims, aft_ims, processed_roidb, delta_bef_roi, delta_aft_roi):
    save_name = '/home/wangshiyao/Documents/testdata/'+processed_roidb[0]['image'].split('/')[-1]
    print 'saving images to '+save_name
    boxes = processed_roidb[0]['boxes']
    ims.squeeze().transpose(1, 2, 0).astype(np.int8)
    bef_ims.squeeze().transpose(1, 2, 0).astype(np.int8)
    aft_ims.squeeze().transpose(1, 2, 0).astype(np.int8)
    delta_bef_roi = np.array(delta_bef_roi).transpose(1, 0, 2)
    delta_aft_roi = np.array(delta_aft_roi).transpose(1, 0, 2)
    for i in range(boxes.shape[0]):
        cv2.rectangle(ims, (int(boxes[i][0]), int(boxes[i][1])),(int(boxes[i][2]), int(boxes[i][3])),(55, 255, 155),5)
        bef_box = bbox_pred(boxes[i].reshape(1, -1), delta_bef_roi[i])
        cv2.rectangle(bef_ims, (int(bef_box[0][0]), int(bef_box[0][1])),(int(bef_box[0][2]), int(bef_box[0][3])),(55, 255, 155),5)
        aft_box = bbox_pred(boxes[i].reshape(1, -1), delta_aft_roi[i])
        cv2.rectangle(aft_ims, (int(aft_box[0][0]), int(aft_box[0][1])),(int(aft_box[0][2]), int(aft_box[0][3])),(55, 255, 155),5)

    imageio.imsave(save_name, ims)
    imageio.imsave(save_name.split('.')[-2]+'_bef'+'.JPEG', bef_ims)
    imageio.imsave(save_name.split('.')[-2]+'_aft'+'.JPEG', aft_ims)

def get_delta_roi(filename, roi_rec, im_scale):
    trackid = roi_rec['gt_trackid']
    boxes = roi_rec['boxes']
    boxes = boxes * im_scale
    delta = np.zeros_like(roi_rec['boxes'], dtype=float)
    dic = {}

    tree = ET.parse(filename)
    size = tree.find('size')
    height = float(size.find('height').text)
    width = float(size.find('width').text)
    objs = tree.findall('object')
    for obj in objs:
        bbox = obj.find('bndbox')
        if roi_rec['flipped']==False:
            np.minimum(float(bbox.find('ymax').text), roi_rec['height']-1)
            dic[int(obj.find('trackid').text)] = [np.maximum(float(bbox.find('xmin').text), 0)*im_scale,
                                         np.maximum(float(bbox.find('ymin').text), 0)*im_scale,
                                         np.minimum(float(bbox.find('xmax').text), roi_rec['width']-1)*im_scale,
                                         np.minimum(float(bbox.find('ymax').text), roi_rec['height']-1)*im_scale]
        else:
            xmin = np.maximum(float(bbox.find('xmin').text), 0)
            ymin = np.maximum(float(bbox.find('ymin').text), 0)
            xmax = np.minimum(float(bbox.find('xmax').text), roi_rec['width']-1)
            ymax = np.minimum(float(bbox.find('ymax').text), roi_rec['height']-1)

            xmin_flip = width - 1 - xmax
            xmax_flip = width - 1 - xmin

            assert xmax_flip >= xmin_flip

            dic[int(obj.find('trackid').text)] = [xmin_flip*im_scale, ymin*im_scale, xmax_flip*im_scale, ymax*im_scale]


    for i in range(len(trackid)):
        if trackid[i] in dic:
            delta_trans = bbox_transform(np.array([boxes[i]]), np.array([dic[trackid[i]]]))
            delta[i][:] = delta_trans[0]
    return delta

def get_triple_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_bef_ims = []
    processed_aft_ims = []
    processed_roidb = []
    processed_delta_bef_roi = []
    processed_delta_aft_roi = []

    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        bef_image = ''
        aft_image = ''
        if roi_rec.has_key('pattern'):
            # get two different frames from the interval [frame_id + MIN_OFFSET, frame_id + MAX_OFFSET]
            offsets = np.random.choice(config.TRAIN.MAX_OFFSET - config.TRAIN.MIN_OFFSET + 1, 2, replace=False) + config.TRAIN.MIN_OFFSET
            bef_id = min(max(roi_rec['frame_seg_id'] + offsets[0], 0), roi_rec['frame_seg_len']-1)
            aft_id = min(max(roi_rec['frame_seg_id'] + offsets[1], 0), roi_rec['frame_seg_len']-1)
            bef_image = roi_rec['pattern'] % bef_id
            aft_image = roi_rec['pattern'] % aft_id

            assert os.path.exists(bef_image), '%s does not exist'.format(bef_image)
            assert os.path.exists(aft_image), '%s does not exist'.format(aft_image)
            bef_im = cv2.imread(bef_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            aft_im = cv2.imread(aft_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            bef_im = im.copy()
            aft_im = im.copy()

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            bef_im = bef_im[:, ::-1, :]
            aft_im = aft_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        bef_im, im_scale = resize(bef_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        aft_im, im_scale = resize(aft_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        bef_im_tensor = transform(bef_im, config.network.PIXEL_MEANS)
        aft_im_tensor = transform(aft_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_bef_ims.append(bef_im_tensor)
        processed_aft_ims.append(aft_im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        new_rec['occluded'] = roi_rec['occluded']

        delta_bef = np.zeros_like(new_rec['boxes'], dtype=float)
        delta_aft = np.zeros_like(new_rec['boxes'], dtype=float)
        if roi_rec.has_key('pattern'):
            bef_annotation = bef_image.replace('Data', 'Annotations').replace('.JPEG','.xml')
            aft_annotation = aft_image.replace('Data', 'Annotations').replace('.JPEG','.xml')
            delta_bef = get_delta_roi(bef_annotation, roi_rec, im_scale)
            delta_aft = get_delta_roi(aft_annotation, roi_rec, im_scale)
        processed_roidb.append(new_rec)
        processed_delta_bef_roi.append(delta_bef)
        processed_delta_aft_roi.append(delta_aft)

    #check_movements(im, bef_im, aft_im, processed_roidb, processed_delta_bef_roi, processed_delta_aft_roi)
    return processed_ims, processed_bef_ims, processed_aft_ims, processed_roidb, processed_delta_bef_roi, processed_delta_aft_roi

def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor

def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor

def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
