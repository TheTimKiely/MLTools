import os, tarfile, sys
from PIL import Image
import numpy as np
import h5py

def read_gz(path):
    print(f'Opening {path}')
    with tarfile.open(path, "r:gz") as tar:
        for tarinfo in tar:
            print(tarinfo)

def create_h5(imgs, name, dir, labels_path, img_size):
    with h5py.File('D:\code\ML\RembrandtML\data\\test_catvnoncat.h5', 'r') as df:
        print(df.keys())

    with h5py.File(name, 'w') as hf:
        #grp = hf.create_group('classes')
        #"test_set_x": shape (50, 64, 64, 3), type "|u1"
        #ds = grp.create_dataset('classes_ds', shape=(2,), dtype='S10')
        #grp = hf.create_group('labels')
        #grp = hf.create_group('data')
        classes = ('not-bloom','bloom')
        classes_encoded = [n.encode('ascii', 'ignore') for n in ('bloom', 'not-bloom')]
        ds_classes = hf.create_dataset('classes', shape=(2,), dtype='|S10', data=classes_encoded)
        #ds_classes.value[0] = 'bloom'
        #ds_classes.value[1] = 'not-bloom'
        ds_classes.flush()
        with open(labels_path, 'r') as labels_fh:
            labels = labels_fh.readlines()
        ds_labels = hf.create_dataset('labels', shape=(len(labels),), dtype=np.dtype('bool'))
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        ds_data = hf.create_dataset('data', shape=(len(labels), *img_size, 3), dtype='|u1')
        i = 0
        for label in labels:
            file_name, l = label.split(',')
            label_val = int(l)
            ds_labels[i] = label_val
            img_path = os.path.join(dir, classes[label_val], file_name)
            print(f'writing {file_name}')
            data = np.reshape(imgs[file_name], (*img_size, 3))
            ds_data[i] = data
            ds_data.flush()
            i = i+1

        hf.flush()

        #hf['classes'][0] = 'bloom'
        #hf['classes'][1] = 'not-bloom'
        #hf['labels'] = '1,1,1,1,1,1,0,0,0,0,'


def resize(imgs, img_path, out_filename, out_path, height, width):
    img = Image.open(img_path)
    img = img.resize((width, height), Image.ANTIALIAS)
    #img.thumbnail()
    try:
        img = img.convert('RGB')
        # creates ((h*w), 3) 2d array
        imgs[out_filename] = list(img.getdata())
        img.save(out_path)
    except Exception as e:
        print(f'Error saving {img_path} as {out_path}\n{e}')
    finally:
        img.close()

if __name__ == '__main__':
    #read_gz('D:\code\ML\RembrandtML\\rembrandtml\model_implementations\MNIST-data\\t10k-images-idx3-ubyte.gz')
    if len(sys.argv) < 2:
        print('Usage:\n\tParameters: dir <path to base directory of images> resize <height and widght '
              'if images are to be resized>')
        base_dir = 'D:\code\ML\projects\cyanotracker\images'
    else:
        if 'dir' in sys.argv:
            base_dir = sys.argv[sys.argv.index('dir')+1]
        if 'resize' in sys.argv:
            size = sys.argv[sys.argv.index('resize')+1]
            img_size = tuple([int(n) for n in size.split(',')])
        else:
            img_size = (128, 128)
    raw_dir = os.path.join(base_dir, 'raw')
    img_index = 1
    label_map = {'bloom': 1, 'not-bloom': 0}
    release_dir = os.path.join(base_dir, 'dataset')


    for ds_dir in os.listdir(raw_dir):
        labels_path = os.path.join(release_dir, ds_dir, 'labels.txt')
        h5_path = os.path.join(release_dir, ds_dir, f'hab_{ds_dir}.h5')
        if os.path.isfile(labels_path):
            os.remove(labels_path)
        imgs = {}

        for imgs_dir in os.listdir(os.path.join(raw_dir, ds_dir)):
            file_dir = os.path.join(raw_dir, ds_dir, imgs_dir)
            out_dir = out_path = os.path.join(release_dir, ds_dir)

            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            for img_file in os.listdir(file_dir):
                img_path = os.path.join(file_dir, img_file)
                out_file = f'{img_index}.jpg'
                out_path = os.path.join(out_dir, out_file)
                resize(imgs, img_path, out_file, out_path, *img_size)
                with open(labels_path, 'a') as f:
                    f.write(f'{out_file},{label_map[imgs_dir]}\n')
                img_index = img_index + 1

            create_h5(imgs, h5_path, release_dir, labels_path, img_size)