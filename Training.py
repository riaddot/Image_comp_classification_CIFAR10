import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchmetrics import Accuracy


from Classifieur import Network 
from Decoder import Generator
from Encoder import Encoder
import channel, instance
import channel, instance
from utils import Data_load,get_labels, test_load
device = torch.device("cuda")



DATA_PATH = r"C:\Users\TEMMMAR\Desktop\PFE\classifier\all_data.npy"     # .np file containing the 64x64 cifar images 
LABEL_PATH = r"C:\Users\TEMMMAR\Desktop\PFE\classifier\all_labels.npy"  # .np file containing the 50000 labels 
CHKP_PATH = r"C:\Users\TEMMMAR\Desktop\Hifi_local\Chekpoint\hific-high.pt"  # checkpoint of the Hifi compressor 
TEST_PATH = r"C:\Users\TEMMMAR\Desktop\PFE\classifier\test_batch.npy"
TEST_LABEL_PATH = r"C:\Users\TEMMMAR\Desktop\PFE\classifier\test_labels.npy"

accuracy = Accuracy(task="multiclass", num_classes=10).to(device=device)







def loss_fn(y_pred,y_hat):
    loss = nn.CrossEntropyLoss().cuda()
    output = loss(y_pred,y_hat)
    return output



def load_pretrained_encoder(path):
    
    load = torch.load(path)
    new_state_dictE = {}
    for name, weight in load['model_state_dict'].items():
        if 'Encoder' in name:
            new_state_dictE[name] = weight

    new_state_dictE1 = {}
    for key, value in new_state_dictE.items():
        new_key = key.replace("Encoder.", "")
        new_state_dictE1[new_key] = value
    
    input_torch = torch.rand((1,3,64,64))
    B = input_torch.shape[0]
    x_dims = tuple(input_torch.size())
    model= Encoder(image_dims=x_dims[1:], C=220)
    model.load_state_dict(new_state_dictE1,strict=False)

    return model 


def train_one_epoch(optimizer,model1,model2,training_loader,epoch_index, tb_writer):
    last_loss = 0.
    running_loss = 0.
    avg_loss = 0.
    last_accuracy = 0
    avg_accuracy = 0 

    for i, data in enumerate(training_loader):
        inputs, label= data
        inputs = inputs.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()

        for param in model1.parameters():
            param.requires_grad = False 

        outputs1 = model1(inputs)
        outputs1 = model2(outputs1)

        
        loss = loss_fn(outputs1,label)
        loss.backward()

        optimizer.step()
        outputs1 = outputs1.data.max(dim=1,keepdim=True)[1]
        outputs1 = outputs1.view(-1).to(device)

        last_accuracy += accuracy(outputs1, label)

        running_loss += loss.item()
        if i % 100 == 0:
            avg_loss += running_loss / 100 # loss per batch
            avg_accuracy += last_accuracy / 100 
            
            avg_loss /= 2
            avg_accuracy /= 2

            print('  batch {} loss: {:.2f} acc:{:.4f}'.format(i + 1, avg_loss,avg_accuracy))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', avg_loss, tb_x)
            tb_writer.add_scalar('accuracy/train', avg_accuracy, tb_x)
            running_loss = 0.
            last_accuracy = 0 
            
    return avg_loss, avg_accuracy

   

def testing(model1,model2,test_loader,tb_writer,epoch_index): 
    trunning_loss = 0.
    tavg_loss = 0.
    t_last_accuracy = 0
    tavg_accuracy = 0  
    model1.eval()
    model2.eval()

    for i, data in enumerate(test_loader):

        inputs, label = data
        inputs = inputs.to(device)
        label = label.to(device)

        model1.eval()
        model2.eval() 

        outputs1t = model1(inputs)
        outputs1t = model2(outputs1t)
       
        losst = loss_fn(outputs1t,label)

        
        outputs1t = outputs1t.data.max(dim=1,keepdim=True)[1]
        outputs1t = outputs1t.view(-1).to(device)
        t_last_accuracy += accuracy(outputs1t, label)

        trunning_loss += losst.item()

        if i % 100 == 0:
            tavg_loss += trunning_loss / 100 # loss per batch
            tavg_accuracy += t_last_accuracy / 100 
            
            tavg_loss /= 2
            tavg_accuracy /= 2

            print('test :   batch {} loss: {:.2f} acc:{:.4f}'.format(i , tavg_loss,tavg_accuracy))
            tb_x = epoch_index*len(test_loader) + i + 1
            tb_writer.add_scalar('Loss/test', tavg_loss, tb_x)
            tb_writer.add_scalar('accuracy/test', tavg_accuracy, tb_x)
            trunning_loss = 0.
            t_last_accuracy = 0 
            
    return tavg_loss, tavg_accuracy

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')





if __name__ == "__main__": 
    

    print("loading data...")
    data_loader,x_dims= Data_load(DATA_PATH)
    test_loader = test_load(TEST_PATH)
    

    B= 2
    model = load_pretrained_encoder(CHKP_PATH)
    model = model.to(device)
    model2 = Network()    
    model2 = model2.to(device)


    A=1
    EPOCHS = 30
    best_val_acc = 0

    a= 1 

    # for fil in filters : 
    print('creating models and optimizers...')

    sgd_optimizer = torch.optim.SGD(model2.parameters(), lr=0.00045, momentum=0.85)
    adam_optimizer = torch.optim.Adam(model2.parameters(), lr=0.0005)
    optimizers = [sgd_optimizer,adam_optimizer]

    writer = SummaryWriter('runs/models6_{}trainer_{}'.format((a+1),timestamp))



    print('strating trainging...')
    
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
        model2.train(True)
        avg_loss, avg_accuracy = train_one_epoch(optimizers[0],model,model2,data_loader,epoch, writer)
        val_loss, val_acc = testing(model,model2,test_loader,writer,epoch)
        print(" model {} training loss:{:.4f} training acc:{:.4f} best_val ={:.4f}".format((a+1),avg_loss, avg_accuracy,best_val_acc))

        print("val acc:{:.4f}".format(val_acc))


        if best_val_acc < val_acc:
            print("acc imporved from {:.4f} to {:.4f}".format(best_val_acc,val_acc))
            best_val_acc = val_acc
            torch.save(model2.state_dict(),"Classifier6_"+str(a+1)+"_best_val_epoch.ckpt")


# %%
