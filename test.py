#!/usr/bin/python
# coding=utf-8

import sys
import os

caffe_root = '/home/boyun/caffe/'
sys.path.append(caffe_root + 'python')
import caffe

caffe.set_mode_gpu()
from pylab import *
from skimage import io;

io.use_plugin('matplotlib')
os.chdir(caffe_root)

model_def = caffe_root + 'examples/myfile/deploy_vgg16_places365.prototxt'
model_weights = caffe_root + 'examples/myfile/vgg16_places365.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)


def convert_mean(binMean, npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb').read()
    blob.ParseFromString(bin_mean)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    npy_mean = arr[0]
    np.save(npyMean, npy_mean)


binMean = '/home/boyun/caffe/examples/myfile/places365CNN_mean.binaryproto'
npyMean = '/home/boyun/caffe/examples/myfile/mean3.npy'
convert_mean(binMean, npyMean)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(npyMean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

with open('/home/boyun/caffe/examples/myfile/places365_val.txt') as image_list:
    with open('/home/boyun/caffe/examples/myfile/result.txt', 'w') as result:
        #with open('/home/boyun/caffe/examples/myfile/1031.txt', 'w') as result1:
            count_right = 0
            count_all = 0
            while 1:
                list_name = image_list.readline()
                if list_name == '\n' or list_name == '':
                    break
                #f = open('/home/boyun/caffe/examples/myfile/1031.txt', 'w')
                imageroot = '/home/boyun/caffe/examples/myfile/val_256/' + str(list_name[0:26])
                if os.path.exists(imageroot):
                    image = caffe.io.load_image(imageroot)
                    transformed_image = transformer.preprocess('data', image)
                    net.blobs['data'].data[...] = transformed_image
                    net.blobs['data'].reshape(1, 3, 224, 224)
                    output = net.forward()
                    output_prob = net.blobs['prob'].data[0]
                    true_label = str(list_name[27:])
                    print str(list_name[0:26]) + ' 真实: ' + str(true_label) + ' 预测: ' + str(output_prob.argmax())
                    count_all += 1
                    #t = 'Places365_val_' + str("%08d" % count_all) + '.jpg '
                    if (int(true_label) == int(str(output_prob.argmax()))):
                        count_right += 1
                        result.writelines(list_name[0:-1]  + ' 预测: ' + str(output_prob.argmax()) + ' Right!'+'\n')
                        #print count_right
                    else:
                        print str(list_name[0:26]) + 'Wrong!'
                        result.writelines(list_name[0:-1]  + ' 预测: ' + str(output_prob.argmax()) + ' Wrong!'+'\n')
                        # os.remove(imageroot)

                    # count_all += 1
                    #t='Places365_val_' + str("%08d" % count_all) + '.jpg '
                    #result.writelines(t + str(list_name[26:])+ '预测: '+str(output_prob.argmax())+'\n')
                    #result1.writelines(t + str(list_name[27:]))
                    if (count_all % 100 == 0):
                        print count_all
                    #path='/home/boyun/caffe/examples/myfile/val_256/'
                    #os.rename(os.path.join(path+str(list_name[0:26])), os.path.join(path, 'Places365_val_'+str("%08d" % count_all) +'.jpg'))
                    #f=open('/home/boyun/caffe/examples/myfile/1031.txt','w')


                else:
                    print 'There is no image!'

                #f.write('\nPlaces365_val_' + str("%07d" % count_all) + '.jpg ' + str(list_name[26:]))

    print 'Accuracy: ' + str(float(count_right) / float(count_all))

    print 'count_all: ' + str(count_all)

    print 'count_right: ' + str(count_right)

    print 'count_wrong: ' + str(count_all - count_right)
