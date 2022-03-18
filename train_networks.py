from os import stat
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
from Memmap import Memmap
import random
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        n_first_O = 30
        n_second_O = 25
        n_third_O = 25
        self.w1_O = nn.Parameter(torch.randn((n_first_O, n_second_O))/1000)
        self.b1_O = nn.Parameter(torch.randn(n_second_O)/1000)
        self.w2_O = nn.Parameter(torch.randn((n_second_O, n_third_O))/1000)
        self.b2_O = nn.Parameter(torch.randn(n_third_O)/1000)
        self.w3_O = nn.Parameter(torch.randn((n_third_O, 1))/1000)
        self.b3_O = nn.Parameter(torch.randn(1)/1000)
        
        n_first_H = 24
        n_second_H = 25
        n_third_H = 25
        self.w1_H = nn.Parameter(torch.randn((n_first_H, n_second_H)) / 1000)
        self.b1_H = nn.Parameter(torch.randn(n_second_H) / 1000)
        self.w2_H = nn.Parameter(torch.randn((n_second_H, n_third_H)) / 1000)
        self.b2_H = nn.Parameter(torch.randn(n_third_H) / 1000)
        self.w3_H = nn.Parameter(torch.randn((n_third_H, 1)) / 1000)
        self.b3_H = nn.Parameter(torch.randn(1) / 1000)
    def forward(self, x_O, x_H, dx_OO, dx_HO, dx_OH, dx_HH):
        z1_O = torch.matmul(x_O, self.w1_O) + self.b1_O
        z2_O = torch.matmul(torch.tanh(z1_O), self.w2_O) + self.b2_O

        z1_H = torch.matmul(x_H, self.w1_H) + self.b1_H
        z2_H = torch.matmul(torch.tanh(z1_H), self.w2_H) + self.b2_H

        ap1_OO = torch.matmul(dx_OO, self.w1_O) / torch.cosh(z1_O) ** 2
        ap2_OO = torch.matmul(ap1_OO, self.w2_O) / torch.cosh(z2_O) ** 2
        y_OO = torch.matmul(ap2_OO, self.w3_O)

        ap1_HO = torch.matmul(dx_HO, self.w1_O) / torch.cosh(z1_O) ** 2
        ap2_HO = torch.matmul(ap1_HO, self.w2_O) / torch.cosh(z2_O) ** 2
        y_HO = torch.matmul(ap2_HO, self.w3_O)

        ap1_HH = torch.matmul(dx_HH, self.w1_H) / torch.cosh(z1_H) ** 2
        ap2_HH = torch.matmul(ap1_HH, self.w2_H) / torch.cosh(z2_H) ** 2
        y_HH = torch.matmul(ap2_HH, self.w3_H)

        ap1_OH = torch.matmul(dx_OH, self.w1_H) / torch.cosh(z1_H) ** 2
        ap2_OH = torch.matmul(ap1_OH, self.w2_H) / torch.cosh(z2_H) ** 2
        y_OH = torch.matmul(ap2_OH, self.w3_H)

        y_O = torch.sum(y_OO, axis=(-1, -2)) + torch.sum(y_OH, axis=(-1, -2))
        y_H = torch.sum(y_HO, axis=(-1, -2)) + torch.sum(y_HH, axis=(-1, -2))  # this is like the change of total energy resulted by the change of H
        return y_O, y_H

class Dataset():
    def __init__(self,**data_path):
        self.data_path = data_path
        self.x_data = {"x_O" : None, "x_H" : None}
        self.ftot_data = {"f_O" : None, "f_H" : None}
        self.dx_data = {"dx_OO" : None,
                        "dx_HO" : None,
                        "dx_OH" : None,
                        "dx_HH" : None}
        self.f_data = {"f_O" : None,
                       "f_H" : None}
        # Load data of x_O x_H fH fO
        self.x_data["x_O"] = torch.tensor(np.load(data_path["x_O"])).to(device)
        self.x_data["x_H"] = torch.tensor(np.load(data_path["x_H"])).to(device)
        temp_fO = np.load(data_path["f_O"])[:,:,0,:]
        temp_fH = np.load(data_path["f_H"])[:,:,0,:]
        self.ftot_data["f_O"] = torch.tensor(np.transpose(temp_fO, axes=(2, 0, 1))).to(device)
        self.ftot_data["f_H"] = torch.tensor(np.transpose(temp_fH, axes=(2, 0, 1))).to(device)
        # Count data
        self.config_number = self.x_data["x_O"].shape[0]
        self.oxygen_number = self.x_data["x_O"].shape[1]
        self.hygrogen_number = self.x_data["x_H"].shape[1]

    # Loading data split or total
    def Data_assign(self,load_mode = "np",assign_number = None,assign_index = None):                    # Load mode = np or memmap
        if load_mode == "np":
            self.dx_data["dx_OO"] = torch.tensor(np.load(self.data_path["dx_OO"])).to(device)
            self.dx_data["dx_HO"] = torch.tensor(np.load(self.data_path["dx_HO"])).to(device)
            self.dx_data["dx_OH"] = torch.tensor(np.load(self.data_path["dx_OH"])).to(device)
            self.dx_data["dx_HH"] = torch.tensor(np.load(self.data_path["dx_HH"])).to(device)
            self.f_data["f_H"] = self.ftot_data["f_H"].to(device)
            self.f_data["f_O"] = self.ftot_data["f_O"].to(device)
        elif load_mode == "memmap":
            if assign_index == None and assign_number != None:
                Oxygen_sample = random.sample([i for i in range(self.oxygen_number)],assign_number)
                Hydrogen_sample = random.sample([i for i in range(self.hygrogen_number)],assign_number)
            
            elif assign_number == None and assign_index != None:
                Oxygen_sample = assign_index["Oxygen"]
                Hydrogen_sample = assign_index["Hydrogen"]
            
            elif assign_number == None and assign_index == None:
                Oxygen_sample = [i for i in range(self.oxygen_number)]
                Hydrogen_sample = [i for i in range(self.hygrogen_number)]
            
            else:
                raise ValueError("Data assign mode error")
            
            temp_dx_OO = Memmap.Memmap_read(self.data_path["dx_OO"])[:,Oxygen_sample,:,:,:]
            self.dx_data["dx_OO"] = torch.tensor(temp_dx_OO,dtype = torch.float).to(device)
            
            temp_dx_HO = Memmap.Memmap_read(self.data_path["dx_HO"])[:,Hydrogen_sample,:,:,:]
            self.dx_data["dx_HO"] = torch.tensor(temp_dx_HO,dtype = torch.float).to(device)
            
            temp_dx_OH = Memmap.Memmap_read(self.data_path["dx_OH"])[:,Oxygen_sample,:,:,:]
            self.dx_data["dx_OH"] = torch.tensor(temp_dx_OH,dtype = torch.float).to(device)
            
            temp_dx_HH = Memmap.Memmap_read(self.data_path["dx_HH"])[:,Hydrogen_sample,:,:,:]
            self.dx_data["dx_HH"] = torch.tensor(temp_dx_HH,dtype = torch.float).to(device)
            
            self.f_data["f_H"] = self.ftot_data["f_H"][:,Hydrogen_sample,:].to(device)
            self.f_data["f_O"] = self.ftot_data["f_O"][:,Oxygen_sample,:].to(device)
        else:
            raise ValueError("Wrong load mode !")

    # Train data generator
    def Data_generator(self):
        for index in range(self.config_number):
            x_O_t = self.x_data["x_O"][index]
            x_H_t = self.x_data["x_H"][index]
            dx_OO_t = self.dx_data["dx_OO"][index]
            dx_HO_t = self.dx_data["dx_HO"][index]
            dx_OH_t = self.dx_data["dx_OH"][index]
            dx_HH_t = self.dx_data["dx_HH"][index]
            fH_t = self.f_data["f_H"][index]
            fO_t = self.f_data["f_O"][index]
            yield(x_O_t,x_H_t,dx_OO_t,dx_HO_t,dx_OH_t,dx_HH_t,fH_t,fO_t)
       
class train_force_BP():
    @staticmethod
    def Train(train_dataset,valid_dataset,train_info_path):
        network = BPNet().to(device)
        optimizer = optim.Adam(network.parameters())
        
        # save parameter
        min_valid_loss = 999
        min_valid_epoch = 0
        train_info = open(train_info_path,"w")
        print("Start epoch !")
        for iepoch in range(200):
            train_dataset.Data_assign(load_mode = "np",assign_number=None)
            valid_dataset.Data_assign(load_mode = "np",assign_number=None)
            
            data_generator = train_dataset.Data_generator()
            valid_data_generator = valid_dataset.Data_generator()
            
            epoch_loss = 0
            epoch_loss_ = 0
            valid_loss = 0
            
            for x_O_t,x_H_t,dx_OO_t,dx_HO_t,dx_OH_t,dx_HH_t,fH_t,fO_t in data_generator:
                optimizer.zero_grad()  # zero the gradient buffers
                yO_pred, yH_pred = network(x_O_t, x_H_t, dx_OO_t, dx_HO_t, dx_OH_t, dx_HH_t)
                loss = torch.sum(torch.abs(yO_pred - fO_t)) + torch.sum(torch.abs(yH_pred - fH_t))
                loss_ =  torch.mean(torch.abs(yO_pred - fO_t)) + torch.mean(torch.abs(yH_pred - fH_t))
                epoch_loss += loss
                epoch_loss_ += loss_
                loss.backward()
                optimizer.step()
            print(yO_pred[:3,:])
            print(fO_t[:3,:])
            epoch_loss /= train_dataset.config_number
            epoch_loss_ /= train_dataset.config_number
            
            for x_O_t,x_H_t,dx_OO_t,dx_HO_t,dx_OH_t,dx_HH_t,fH_t,fO_t in valid_data_generator:
                yO_pred, yH_pred = network(x_O_t, x_H_t, dx_OO_t, dx_HO_t, dx_OH_t, dx_HH_t)
                loss = torch.mean(torch.abs(yO_pred - fO_t)) + torch.mean(torch.abs(yH_pred - fH_t))
                valid_loss += loss
            valid_loss /= valid_dataset.config_number
            
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                min_valid_epoch = iepoch
                torch.save(network.state_dict(), "test.pth")
            
            print("{}\t{}\t{}\t{}".format(iepoch,epoch_loss,epoch_loss_,valid_loss))
            train_info.write("{} {} {} {}\n".format(iepoch,epoch_loss,epoch_loss_,valid_loss))
            train_info.flush()
            
        print("Min_valid_epoch : {} , Min_valid_loss : {} ".format(min_valid_epoch,min_valid_loss))
    @staticmethod
    def Load_model(model_path):
        network = BPNet()
        network.load_state_dict(torch.load(model_path))
        return(network)
    @staticmethod
    def Test_data(dataset,model_path):
        total_loss = {"O" : [],"H" : []}
        network = train_force_BP.Load_model(model_path)
        data_generator = dataset.Data_generator()
        for x_O_t,x_H_t,dx_OO_t,dx_HO_t,dx_OH_t,dx_HH_t,fH_t,fO_t in tqdm(data_generator):
            yO_pred, yH_pred = network(x_O_t, x_H_t, dx_OO_t, dx_HO_t, dx_OH_t, dx_HH_t)
            error_matrix_O = torch.abs(yO_pred - fO_t)
            error_matrix_H = torch.abs(yH_pred - fH_t)
            total_loss["O"].append(error_matrix_O.detach().numpy())
            total_loss["H"].append(error_matrix_H.detach().numpy())
        return(total_loss)

def Test_result_analysis(total_loss,O_coord_path,H_coord_path,save_path,save_title):
    O_coord_matrix = np.load(O_coord_path)[:,:,0,:]
    O_coord_matrix = np.transpose(O_coord_matrix, axes=(2,0,1))
    H_coord_matrix = np.load(H_coord_path)[:,:,0,:]
    H_coord_matrix = np.transpose(H_coord_matrix, axes=(2,0,1))

    O_error_matrix = np.array(total_loss["O"])
    H_error_matrix = np.array(total_loss["H"])

    Oxygen_data = {"data":O_coord_matrix[:,:,2].flatten(),"error" : O_error_matrix.mean(axis = 2).flatten()}
    Hydrogen_data = {"data" : H_coord_matrix[:,:,2].flatten(),"error" : H_error_matrix.mean(axis=2).flatten()}

    with open("{}/{}_O.dat".format(save_path,save_title),"w") as O_error:
        for index in range(Oxygen_data["data"].shape[0]):
            O_error.write("{} {}\n".format(Oxygen_data["data"][index],Oxygen_data["error"][index]))
    with open("{}/{}_H.dat".format(save_path,save_title),"w") as H_error:
        for index in range(Hydrogen_data["data"].shape[0]):
            H_error.write("{} {}\n".format(Hydrogen_data["data"][index],Hydrogen_data["error"][index]))
        

if __name__ == "__main__":

    bulk_train_data_path = "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Train_data/Bulk_train_data/Train_data-30/"
    bulk_valid_data_path = "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Valid_data/Bulk_Water_test"
    train_dataset = Dataset(x_O = "{}/xO.npy".format(bulk_train_data_path),
                            x_H = "{}/xH.npy".format(bulk_train_data_path),
                            dx_OO = "{}/xOO_d.npy".format(bulk_train_data_path),
                            dx_HO = "{}/xHO_d.npy".format(bulk_train_data_path),
                            dx_OH = "{}/xOH_d.npy".format(bulk_train_data_path),
                            dx_HH = "{}/xHH_d.npy".format(bulk_train_data_path),
                            f_H = "{}/fH_allconfigs.npy".format(bulk_train_data_path),
                            f_O = "{}/fO_allconfigs.npy".format(bulk_train_data_path))

    valid_dataset = Dataset(x_O = "{}/xO.npy".format(bulk_valid_data_path),
                            x_H = "{}/xH.npy".format(bulk_valid_data_path),
                            dx_OO = "{}/xOO_d.npy".format(bulk_valid_data_path),
                            dx_HO = "{}/xHO_d.npy".format(bulk_valid_data_path),
                            dx_OH = "{}/xOH_d.npy".format(bulk_valid_data_path),
                            dx_HH = "{}/xHH_d.npy".format(bulk_valid_data_path),
                            f_H = "{}/fH_allconfigs.npy".format(bulk_valid_data_path),
                            f_O = "{}/fO_allconfigs.npy".format(bulk_valid_data_path))


    train_force_BP.Train(train_dataset,valid_dataset,"./train_info_force_BP_yihao.txt")

    # bulk_test_data = "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Valid_data/Bulk_Water_test"
    # lv_test_data = "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Valid_data/Liquid_Vaper_test"
    # test_data_path = bulk_test_data
    # test_dataset = Dataset( x_O = "{}/xO.npy".format(test_data_path),
    #                         x_H = "{}/xH.npy".format(test_data_path),
    #                         dx_OO = "{}/xOO_d.npy".format(test_data_path),
    #                         dx_HO = "{}/xHO_d.npy".format(test_data_path),
    #                         dx_OH = "{}/xOH_d.npy".format(test_data_path),
    #                         dx_HH = "{}/xHH_d.npy".format(test_data_path),
    #                         f_H = "{}/fH_allconfigs.npy".format(test_data_path),
    #                         f_O = "{}/fO_allconfigs.npy".format(test_data_path))
    
    # test_dataset.Data_assign(load_mode="np")
    # total_loss = train_force_BP.Test_data(  test_dataset,
    #                                         "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Model_result/trained_force_model_statedict_BP_bulkH2O.pth")
    # Test_result_analysis(   total_loss,
    #                         "{}/Oxyz_allconfigs.npy".format(test_data_path),
    #                         "{}/Hxyz_allconfigs.npy".format(test_data_path),
    #                         save_path = "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Data/Model_result/",
    #                         save_title ="Bulk_test")















