from os import stat
import numpy as np
from numpy.lib.function_base import quantile
from tqdm import tqdm
from glob import glob
# from parameters import *
from Memmap import *
import random

natoms = 192
noxygen = 64
nhydrogen = 128

# natoms = 3072
# noxygen = 1024
# nhydrogen = 2048
nfolders = 1

"""
    This script is used to merge features in each config file
    Reference : scfnn(https://github.com/andy90/SCFNN)


"""


class Assemble_config():
    @staticmethod
    def Reading_single_file(data_path):
        with open(data_path,"r") as f:
            lines = f.readlines()
            lines = [line.replace("\n","") for line in lines]
            natom = int(lines[0].split()[0])
            box_range = [float(i) for i in lines[1].split()[1:]]
            data = np.zeros((natom,3))
            lines = lines[2:]
            for index,line in enumerate(lines):
                data[index,:] = np.array([float(i) for i in line.split()[1:]])
        return(natom,np.array(box_range),data)
    @staticmethod
    def Assemble_basic_data(save_folder_path,config_list,assemble_data_save_path,config_name = "config.bulk.xyz",force_name = "force.bulk.xyz",atom_unit = True):
        Oxyz_all = ()       # the tuple that stores all the Oxygen xyz
        Hxyz_all = ()       # the tuple that stores all the Hydrogen xyz
        fO_all = ()         # the tuple that stores all the Oxygen force
        fH_all = ()         # the tuple that stores all the Hydrogen force
        boxlength_all = ()  # the tuple that stores all the boxlengths
        
        # config_paths = list(glob("{}/*".format(save_folder_path)))
        
        # config_paths = []
        # for data_file_path in folder_names:
        #     for file_name in good_configs:
        #         config_paths.append("{}/{}".format(data_file_path,file_name))
        config_paths = ["{}/{}".format(save_folder_path,i) for i in config_list]

        natom = 0
        nconfigs = len(config_paths)
        for config_path in tqdm(config_paths):
            config_file_path = "{}/{}".format(config_path,config_name)
            force_file_path = "{}/{}".format(config_path,force_name)
             # read in the coord from files
            natom,box_range,config_matrix = Assemble_config.Reading_single_file(config_file_path)
            # read in the forces from files
            _,_,force_matrix = Assemble_config.Reading_single_file(force_file_path)
            # change the unit from Angstrom to Bohr
            if atom_unit:
                box_range /= 0.529177
                # config_matrix /= 0.529177
                # force_matrix *= (1e-4 / (6.02 * 8.2387234983))        # force unit from 10J/(mol*A) to a.u. force unit
            # enforce pbc, the center of the box is 0
            config_matrix = config_matrix - np.round(config_matrix / box_range) * box_range  
            # extract the coordinates of the oxygen and hydrogen
            Oxygen_index = np.arange(0,natom,3)
            Hydrogen_index = np.sort(np.concatenate((np.arange(1, natom-1, 3), np.arange(2, natom, 3))))

            Oxygen_xyz = config_matrix[Oxygen_index,:]
            Hydrogen_xyz = config_matrix[Hydrogen_index,:]

            Oxygen_force = force_matrix[Oxygen_index,:]
            Hydrogen_force = force_matrix[Hydrogen_index,:]
            # Save config data
            np.savetxt("{}/Oxyz.txt".format(config_path), Oxygen_xyz)
            np.savetxt("{}/Hxyz.txt".format(config_path), Hydrogen_xyz)
            np.savetxt("{}/box.txt".format(config_path), box_range)
            np.savetxt("{}/fO.txt".format(config_path), Oxygen_force)
            np.savetxt("{}/fH.txt".format(config_path), Hydrogen_force)
            # Assemble data
            Oxyz_all += (Oxygen_xyz,)
            Hxyz_all += (Hydrogen_xyz, )
            fO_all += (Oxygen_force,)
            fH_all += (Hydrogen_force,)
        noxygen = int(natom/3)
        nhydrogen = natom - noxygen
        Oxyz_stack = np.stack(Oxyz_all, axis=-1).reshape((noxygen, 3, nfolders,nconfigs))
        Hxyz_stack = np.stack(Hxyz_all, axis=-1).reshape((nhydrogen, 3, nfolders, nconfigs))
        fO_stack = np.stack(fO_all, axis=-1).reshape((noxygen, 3, nfolders, nconfigs))
        fH_stack = np.stack(fH_all, axis=-1).reshape((nhydrogen, 3, nfolders, nconfigs))
        
        np.save("{}/Oxyz_allconfigs".format(assemble_data_save_path), Oxyz_stack)
        np.save("{}/Hxyz_allconfigs".format(assemble_data_save_path), Hxyz_stack)
        np.save("{}/fO_allconfigs".format(assemble_data_save_path), fO_stack)
        np.save("{}/fH_allconfigs".format(assemble_data_save_path), fH_stack)
    @staticmethod
    def Assemble_feature(config_folder_path,config_list,save_folder_path,feature_name,id_center):
        if id_center == "O":
            ncenter = noxygen
        else:
            ncenter = nhydrogen
        
        features_all = ()
        features_d_all = ()
        for config_folder_name in tqdm(config_list):
            feature_path = "{}/{}/features_{}.txt".format(config_folder_path,config_folder_name,feature_name)
            feature_d_path = "{}/{}/features_d{}.txt".format(config_folder_path,config_folder_name,feature_name)
            features = np.loadtxt(feature_path, dtype=np.float32)
            dfeatures = np.loadtxt(feature_d_path, dtype=np.float32)
            features_all += (features, )
            features_d_all += (dfeatures.reshape((features.shape[0], ncenter, natoms, 3)), )    
        features_all = np.transpose(np.stack(features_all, axis=0), axes=(0, 2, 1))  # now it is (nconfig, ncenter, nfeatures)
        features_d_all = np.transpose(np.stack(features_d_all, axis=0), axes=(0, 2, 3, 4, 1))  # now it is (nconfig, ncenter, natoms, 3, nfeatures)

        np.save("{}/features_{}".format(save_folder_path,feature_name), features_all)
        np.save("{}/features_d{}".format(save_folder_path,feature_name), features_d_all)   
    @staticmethod
    def Assemble_features_further(feature_file_path,feature_name_tuple):  # assemble features that have the same center atom
        features_all = ()
        features_d_all = ()
        for feature_name in tqdm(feature_name_tuple):
            features = np.load("{}/features_{}".format(feature_file_path,feature_name) + ".npy")
            features_d = np.load("{}/features_d{}".format(feature_file_path,feature_name) + ".npy")
            features_all += (features,)
            features_d_all += (features_d,)

        features_all = np.concatenate(features_all, axis=-1)  # stack along the nfeatures axis
        features_d_all = np.transpose(np.concatenate(features_d_all, axis=-1), axes=(0, 2, 3, 1, 4))
        # stack along the nfeatures axis, then make sure the number of center atoms is at the second last axis

        return features_all, features_d_all
    @staticmethod
    def Generate_train_data(config_folder_path,train_config_list,save_folder_path):
        Assemble_config.Assemble_basic_data(config_folder_path,train_config_list,save_folder_path)

        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2OO","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2OH","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2HH","H")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2HO","H")

        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4OOO","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4OOH","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4OHH","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4HOO","H")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4HHO","H")
        print("Furter assemble !")
        xO,xO_d = Assemble_config.Assemble_features_further(save_folder_path,("G2OO", "G2OH", "G4OOO", "G4OOH", "G4OHH"))
        xH,xH_d = Assemble_config.Assemble_features_further(save_folder_path,("G2HH", "G2HO", "G4HHO", "G4HOO"))

        xOO_d = xO_d[:, :noxygen, :, :, :]  # the effect of the move of O atoms on the O features
        xHO_d = xO_d[:, noxygen:, :, :, :]  # the effect of the move of H atoms on the O features
        xHH_d = xH_d[:, noxygen:, :, :, :]  # the effect of the move of H atoms on the H features
        xOH_d = xH_d[:, :noxygen, :, :, :]  # the effect of the move of O atoms on the H features
        
        xO_av = np.mean(xO, axis=(0, 1))
        xO_min = np.min(xO, axis=(0, 1))
        xO_max = np.max(xO, axis=(0, 1))
        np.savetxt("{}/xO_scalefactor.txt".format(save_folder_path), np.stack((xO_av, xO_min, xO_max), axis=-1))
        
        xH_av = np.mean(xH, axis=(0, 1))
        xH_min = np.min(xH, axis=(0, 1))
        xH_max = np.max(xH, axis=(0, 1))

        np.savetxt("{}/xH_scalefactor.txt".format(save_folder_path), np.stack((xH_av, xH_min, xH_max), axis=-1))
        
        xO = (xO - xO_av) / (xO_max - xO_min)
        xH = (xH - xH_av) / (xH_max - xH_min)
        
        xOO_d = xOO_d / (xO_max - xO_min)
        xHO_d = xHO_d / (xO_max - xO_min)
        xHH_d = xHH_d / (xH_max - xH_min)
        xOH_d = xOH_d / (xH_max - xH_min)
        print("Start Saving :")
        Memmap.Memmap_save(save_folder_path,"xO",xO)
        Memmap.Memmap_save(save_folder_path,"xH",xH)
        Memmap.Memmap_save(save_folder_path,"xOO_d",xOO_d)
        Memmap.Memmap_save(save_folder_path,"xHO_d",xHO_d)
        Memmap.Memmap_save(save_folder_path,"xHH_d",xHH_d)
        Memmap.Memmap_save(save_folder_path,"xOH_d",xOH_d)

        np.save("{}/xO".format(save_folder_path), xO)
        np.save("{}/xH".format(save_folder_path), xH)
        np.save("{}/xOO_d".format(save_folder_path), xOO_d)
        np.save("{}/xHO_d".format(save_folder_path), xHO_d)
        np.save("{}/xHH_d".format(save_folder_path), xHH_d)
        np.save("{}/xOH_d".format(save_folder_path), xOH_d)
    @staticmethod
    def Generate_valid_data(config_folder_path,train_config_list,save_folder_path,xO_scale_path,xH_scale_path):
        Assemble_config.Assemble_basic_data(config_folder_path,train_config_list,save_folder_path)

        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2OO","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2OH","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2HH","H")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2HO","H")

        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4OOO","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4OOH","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4OHH","O")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4HOO","H")
        Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4HHO","H")


        xO,xO_d = Assemble_config.Assemble_features_further(save_folder_path,("G2OO", "G2OH", "G4OOO", "G4OOH", "G4OHH"))
        xH,xH_d = Assemble_config.Assemble_features_further(save_folder_path,("G2HH", "G2HO", "G4HHO", "G4HOO"))

        xOO_d = xO_d[:, :noxygen, :, :, :]  # the effect of the move of O atoms on the O features
        xHO_d = xO_d[:, noxygen:, :, :, :]  # the effect of the move of H atoms on the O features
        xHH_d = xH_d[:, noxygen:, :, :, :]  # the effect of the move of H atoms on the H features
        xOH_d = xH_d[:, :noxygen, :, :, :]  # the effect of the move of O atoms on the H features
        
        xO_scale = np.loadtxt(xO_scale_path,dtype = np.float32)
        xH_scale = np.loadtxt(xH_scale_path,dtype = np.float32)

        xO = (xO - xO_scale[:,0]) / (xO_scale[:,2] - xO_scale[:,1])
        xH = (xH - xH_scale[:,0]) / (xH_scale[:,2] - xH_scale[:,1])
        
        xOO_d = xOO_d / (xO_scale[:,2] - xO_scale[:,1])
        xHO_d = xHO_d / (xO_scale[:,2] - xO_scale[:,1])
        xHH_d = xHH_d / (xH_scale[:,2] - xH_scale[:,1])
        xOH_d = xOH_d / (xH_scale[:,2] - xH_scale[:,1])
        print("Start Saving :")
        Memmap.Memmap_save(save_folder_path,"xO",xO)
        Memmap.Memmap_save(save_folder_path,"xH",xH)
        Memmap.Memmap_save(save_folder_path,"xOO_d",xOO_d)
        Memmap.Memmap_save(save_folder_path,"xHO_d",xHO_d)
        Memmap.Memmap_save(save_folder_path,"xHH_d",xHH_d)
        Memmap.Memmap_save(save_folder_path,"xOH_d",xOH_d)

        np.save("{}/xO".format(save_folder_path), xO)
        np.save("{}/xH".format(save_folder_path), xH)
        np.save("{}/xOO_d".format(save_folder_path), xOO_d)
        np.save("{}/xHO_d".format(save_folder_path), xHO_d)
        np.save("{}/xHH_d".format(save_folder_path), xHH_d)
        np.save("{}/xOH_d".format(save_folder_path), xOH_d)


if __name__ == "__main__":
    configs_data_path = "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Origin_data/Bulk64_Yihao/"

    Assemble_config.Generate_train_data(configs_data_path,
                                        [i for i in range(1000)],
                                        "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Train_data/Bulk_Yihao_data/")

    Assemble_config.Generate_valid_data(configs_data_path,
                                        [i for i in range(1000,1500)],
                                        "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Valid_data/Bulk_Yihao_data",
                                        "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Train_data/Bulk_Yihao_data/xO_scalefactor.txt",
                                        "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Train_data/Bulk_Yihao_data/xH_scalefactor.txt")

    # train_config_list = random.sample([i for i in range(280)],30)

    # valid_config_list = random.sample([i for i in range(280,300)],5)
    # print(valid_config_list)

    # train_config_list = [130, 123, 27, 152, 169, 276, 146, 94, 134, 226, 46, 76, 85, 25, 277, 13, 30, 66, 21, 20, 83, 217, 202, 12, 55, 207, 173, 68, 224, 190]
    # train_config_list = [130, 123, 27, 152, 169, 276, 146, 94, 134, 226, 46, 76, 85, 25, 277, 13, 30, 66, 21, 20]


    # liquid_vaper_test_list = [146, 96, 119, 3, 47]
    # # liquid_vaper_test_list = random.sample([i for i in range(0,170)],5)
    # print(liquid_vaper_test_list)

    # Assemble_config.Generate_valid_data(configs_data_path,
    #                                     liquid_vaper_test_list,
    #                                     "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Valid_data/Liquid_Vaper_test",
    #                                     "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Train_data/Bulk_train_data/Train_data-30/xO_scalefactor.txt",
    #                                     "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Train_data/Bulk_train_data/Train_data-30/xH_scalefactor.txt")

    # Assemble_config.Generate_valid_data(configs_data_path,
    #                                     valid_config_list,
    #                                     valid_data_path,
    #                                     "/DATA/users/yanghe/projects/BPNN_project/Data/Bulk_data/Train_data/xO_scalefactor.txt",
    #                                     "/DATA/users/yanghe/projects/BPNN_project/Data/Bulk_data/Train_data/xH_scalefactor.txt")

