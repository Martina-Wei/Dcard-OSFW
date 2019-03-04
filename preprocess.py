import os
import argparse

def get_dir_info(dir_path):
    
    list_IDs = []
    labels = {}
    labels_num = []
    
    list_Labels = sorted(os.listdir(dir_path))
    img_nums = []
    
    for subdir in list_Labels:
        subdir_fns = os.listdir(os.path.join(dir_path, subdir))
        img_nums.append(len(subdir_fns))

        for subdir_fn in subdir_fns:
            list_IDs.append(subdir_fn)
            labels_num.append(list_Labels.index(subdir))
            labels[subdir_fn] = subdir

    res = {
        'list_IDs': list_IDs,
        'img_nums': img_nums,
        'list_Labels': list_Labels
    }
    return res

def prepare_train_valid_folder(src_dir, des_path):
    cp_train_cmd = 'cp -r %s %s/train'%(src_dir, des_path)
    os.system(cp_train_cmd)
    
    cp_train_cmd = 'cp -r %s %s/valid'%(src_dir, des_path)
    os.system(cp_train_cmd)
    
    ori_dir_info = get_dir_info(src_dir)
    print(ori_dir_info['img_nums'])
    
    train_path = os.path.join(des_path, 'train')
    valid_path = os.path.join(des_path, 'valid')
    for subdir in ori_dir_info['list_Labels']:
        subdir_fns = sorted(os.listdir(os.path.join(src_dir, subdir)))
        valid_num = len(subdir_fns)//10 + 1
        valid_fns = subdir_fns[:valid_num]
        train_fns = subdir_fns[valid_num:]
        for valid_fn in valid_fns:
            os.remove(os.path.join(train_path, subdir, valid_fn))
        for train_fn in train_fns:
            os.remove(os.path.join(valid_path, subdir, train_fn))
            
    print(get_dir_info(train_path)['img_nums'])
    print(get_dir_info(valid_path)['img_nums'])
    
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--action', help='split data to train & valid', default='split_data', type=str)
    PARSER.add_argument('--src_dir_path', help='source directory path', default=None, type=str)
    PARSER.add_argument('--des_path', help='destination path', default=None, type=str)
    
    ARGS = PARSER.parse_args()
    
    if ARGS.action=='split_data':
        prepare_train_valid_folder(ARGS.src_dir_path, ARGS.des_path)