from distutils.command.config import config
import numpy as np
import math
import os
'''
transform different data type to readable data (between different unit)

                                            Zara Yang
                                            2022/3/18

'''

box_range = np.array([12.41719651,12.41719651,12.41719651])
coord_file_path = "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Traj_data/Bulk64_Liyan/fort.336"
force_file_path = "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Traj_data/Bulk64_Liyan/fort.333"
natom = 192


def pbc(pos_matrix,box_range):
    for index in range(pos_matrix.shape[0]):
        pos_matrix[index,:] = pos_matrix[index,:] - np.round(pos_matrix[index,:]/box_range) * box_range

def Transform_L2R(save_path,coord_file_path,force_file_path,box_range,natom):
    coord_matrix = np.loadtxt(coord_file_path)[:,1:]
    force_matrix = np.loadtxt(force_file_path)[:,1:]
    pbc(coord_matrix,box_range)
    for nconfig in range(coord_matrix.shape[0]//natom):
        config_path = "{}/{}".format(save_path,nconfig)
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        
        box_path = "{}/box.txt".format(config_path)
        coord_path = "{}/config.bulk.xyz".format(config_path)
        force_path = "{}/force.bulk.xyz".format(config_path)

        np.savetxt(box_path,box_range)
        with open(coord_path,"w") as coord_f:
            coord_f.write("{}\n".format(natom))
            coord_f.write("config_in_3d_box   {}       {}       {}\n".format(box_range[0],box_range[1],box_range[2]))
            for center_index in range(nconfig * natom,(nconfig + 1)*natom,3):
                coord_f.write("O\t{} {} {}\n".format(coord_matrix[center_index,0],coord_matrix[center_index,1],coord_matrix[center_index,2]))
                coord_f.write("H\t{} {} {}\n".format(coord_matrix[center_index+1,0],coord_matrix[center_index+1,1],coord_matrix[center_index+1,2]))
                coord_f.write("H\t{} {} {}\n".format(coord_matrix[center_index+2,0],coord_matrix[center_index+2,1],coord_matrix[center_index+2,2]))
        with open(force_path,"w") as force_f:
            force_f.write("{}\n".format(natom))
            force_f.write("config_in_3d_box   {}       {}       {}\n".format(box_range[0],box_range[1],box_range[2]))
            for center_index in range(nconfig * natom,(nconfig + 1)*natom,3):
                force_f.write("O\t{} {} {}\n".format(force_matrix[center_index,0],force_matrix[center_index,1],force_matrix[center_index,2]))
                force_f.write("H\t{} {} {}\n".format(force_matrix[center_index+1,0],force_matrix[center_index+1,1],force_matrix[center_index+1,2]))
                force_f.write("H\t{} {} {}\n".format(force_matrix[center_index+2,0],force_matrix[center_index+2,1],force_matrix[center_index+2,2]))

def Transform_Y2R(save_path,data_file_path,save_title,box_range,natom):
    # =======================================
    # Data unit from Yihao 
    #   coord : a.u.
    #   force : a.u.
    #   box   ：Angstrom
    # ========================================
    config_index = 0
    config_path = ""
    save_handle = None
    with open(data_file_path,"r") as f:
        for line in f:
            line = line.replace("\n","")
            line_list = line.split()
            if len(line_list) == 1:
                if save_handle != None:
                    save_handle.close()
                config_path = "{}/{}".format(save_path,config_index)
                config_index += 1
                if not os.path.exists(config_path):
                    os.makedirs(config_path)
                np.savetxt("{}/box.txt".format(config_path),box_range)
                save_handle = open("{}/{}".format(config_path,save_title),"w")
                save_handle.write("{}\n".format(natom))
                save_handle.write("config_in_3d_box\t{}\t{}\t{}\n".format(box_range[0],box_range[1],box_range[2]))
            elif(len(line_list)> 4):
                continue
            else:
                save_handle.write("{}\n".format(line))

class Data_transfer():
    """
        Transfer CP2K output data to readable file like Hu's file style 
        
                                                        ZaraYang
                                                        2022/3/18
        
    """
    @staticmethod
    def Reading_coord(data_path,start_index = 2):
        coord_matrix = np.zeros((natom,3))
        with open(data_path,"r") as f:
            lines = f.readlines()
            lines = [i.replace("\n","") for i in lines[start_index:]]
            lines = lines[:natom]
            for index,line in enumerate(lines):
                line = line.split()[1:]
                coord_matrix[index,:] = np.array([float(i) for i in line])
        return(coord_matrix)
    @staticmethod
    def Reading_force(data_path,start_index = 4):
        force_matrix = np.zeros((natom,3))
        with open(data_path,"r") as f:
            lines = f.readlines()
            lines = [i.replace("\n","") for i in lines[start_index:]]
            lines = lines[:natom]
            for index,line in enumerate(lines):
                line = line.split()[3:]
                force_matrix[index,:] = np.array([float(i) for i in line])
        return(force_matrix)
    @staticmethod
    def Reading_boxlength(data_path):
        box_length = 0
        with open(data_path,"r") as f:
            lines = f.readlines()
            box_length = lines[1].split()[10]
        return(box_length)
    @staticmethod
    def Transfor_data(config_path,force_file_name = "W64-bulk-W64-forces-1_0.xyz",coord_file_name = "W64-bulk-HOMO_centers_s1-1_0.xyz"):
        force_matrix = Data_transfer.Reading_force("{}/{}".format(config_path,force_file_name))
        coord_matrix = Data_transfer.Reading_coord("{}/{}".format(config_path,coord_file_name))
        box_length = Data_transfer.Reading_boxlength("{}/init.xyz".format(config_path))
        with open("{}/config.bulk.xyz".format(config_path),"w") as coord_f:
            coord_f.write("{}\n".format(natom))
            coord_f.write("config_in_3d_box {} {} {}\n".format(box_length,box_length,box_length))
            for index in range(natom):
                if index % 3 == 0:
                    atom_type = "O"
                else:
                    atom_type = "H"
                coord_f.write("{} {} {} {}\n".format(atom_type, coord_matrix[index,:][0], coord_matrix[index,:][1], coord_matrix[index,:][2]))
        with open("{}/force.bulk.xyz".format(config_path),"w") as force_f:
            force_f.write("{}\n".format(natom))
            force_f.write("config_in_3d_box {} {} {}\n".format(box_length,box_length,box_length))
            for index in range(natom):
                if index % 3 == 0:
                    atom_type = "O"
                else:
                    atom_type = "H"
                force_f.write("{} {} {} {}\n".format(atom_type, force_matrix[index,:][0], force_matrix[index,:][1], force_matrix[index,:][2]))

if __name__ == "__main__":
    # Transform_L2R(  "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Origin_data/Bulk64_Liyan",
    #                 coord_file_path,
    #                 force_file_path,
    #                 box_range,
    #                 192)

    Transform_Y2R(  "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Origin_data/Bulk64_Yihao/",
                    "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Traj_data/Buli64_yihao/waterMD-pos-1.xyz",
                    "config.bulk.xyz",  
                    box_range,
                    192)

    Transform_Y2R(  "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Origin_data/Bulk64_Yihao/",
                    "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Traj_data/Buli64_yihao/waterMD-frc-1.xyz",
                    "force.bulk.xyz",  
                    box_range,
                    192)





