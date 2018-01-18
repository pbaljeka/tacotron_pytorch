import os
import sys
import numpy as np

def calc_mean(indir, filelist, feature_dimension=10):
    temp_sum=np.zeros((int(feature_dimension)))
    num_frames=0
    with open(filelist, 'r') as f:
        for filename in f:
            filepath = os.path.join(indir+filename.strip())
            data_mat = np.load(filepath)
            temp_sum+= np.sum(data_mat, axis=0)
            num_frames += data_mat.shape[0]
    return temp_sum / num_frames

def calc_std(indir, filelist, mean, feature_dimension=10):
    temp_std=np.zeros((int(feature_dimension)))
    num_frames=0
    with open(filelist, 'r') as f:
        for filename in f:
            filepath = os.path.join(indir +filename.strip())
            data_mat = np.load(filepath)
            frames = data_mat.shape[0]
            temp_std=temp_std + np.sum(((data_mat - np.tile(mean,(frames,1)))**2), axis=0)
            num_frames += frames
    return np.sqrt(temp_std / num_frames)

def calc_f0_stats(indir, outdir, filelist, feature_simension=10):
    mean_vec=calc_mean(indir, filelist)
    std_vec = calc_std(indir, filelist, mean_vec)
    print('Saving stats')
    np.save(os.path.join(outdir,'f0_mean'), mean_vec, allow_pickle=False)
    np.save(os.path.join(outdir, 'f0_std'), std_vec, allow_pickle=False)
    return mean_vec, std_vec

def normalize_data(indir, norm_outdir, filelist, mean, std, feature_dimension=10):
    with open(filelist,'r') as f:
        for filename in f:
            print(filename)
            filepath = os.path.join(os.getcwd(), indir, filename.strip())
            outfilepath = os.path.join(os.getcwd(), norm_outdir, filename.strip())
            data_mat= np.load(filepath)
            norm_data= (data_mat - np.tile(mean, (data_mat.shape[0],1)))/ (np.tile(std, (data_mat.shape[0],1)) + 0.00000001)
            np.save(outfilepath, norm_data, allow_pickle=False)
    return 

def denormalize_data(indir, denorm_outdir, filelist, mean, std, feature_dimension=10):
    with open(filelist,'r') as f:
        for filename in f:
            filepath = os.path.join(os.getcwd(), indir, filename.strip())
            outfilepath = os.path.join(os.getcwd(), denorm_outdir, filename.strip())
            data_mat= np.load(filepath)
            denorm_data=  (data_mat* (np.tile(std, (data_mat.shape[0],1))+ 0.00000001)) + np.tile(mean, (data_mat.shape[0],1))
            np.save(outfilepath, denorm_data, allow_pickle=False)
    return 


if __name__=="__main__":
    train_filelist = sys.argv[1]
    test_filelist = sys.argv[2]
    curr_dir = os.getcwd()
    indir = curr_dir + "/training/"
    outdir= curr_dir + "/norm_stats/"
    norm_outdir = curr_dir + "/norm_f0_feats/"
    denorm_outdir = curr_dir + "/denorm_f0_feats/"
    if not os.path.exists(outdir):
        print('Creating norm_stats')
        os.mkdir(outdir)	
    if not os.path.exists(norm_outdir):
       os.mkdir(norm_outdir)
    if not os.path.exists(denorm_outdir):
       os.mkdir(denorm_outdir)	
    f0_mean, f0_std =calc_f0_stats(indir, outdir, train_filelist)
    normalize_data(indir, norm_outdir, train_filelist, f0_mean, f0_std)
    normalize_data(indir, norm_outdir, test_filelist, f0_mean, f0_std)
   
    #denormalize_data(norm_outdir, denorm_outdir, train_filelist, f0_mean, f0_std)
