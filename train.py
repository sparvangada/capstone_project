#original source ref:https://github.com/pyaf/
import time
import copy
import torch
from torchnet import meter
from torch.autograd import Variable
import matplotlib.pyplot as plt



data_cat = ['train', 'valid'] # data categories

def train_model(model, criterion, optimizer, dataloaders, scheduler, 
                dataset_sizes, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x:[] for x in data_cat} # for storing costs per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        confusion_matrix = {x: meter.ConfusionMeter(2, normalized=True) 
                            for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        #print('I am ok so far 1')
        for phase in data_cat:
            print("================Phase={}".format(phase))
            model.train(phase=='train')
            #print("I am ok so far 2")
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                print(i, end='\r')
                inputs = data['images'][0]
                if phase == 'valid': 
                    with torch.no_grad():
                        labels = data['label'].type(torch.FloatTensor)
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())    
                        #print('with floattensor and cuda label={}'.format(labels))
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        outputs = model(inputs)
                        outputs = torch.mean(outputs)
                        #print('output after torch mean={} '.format(outputs))
                        loss = criterion(outputs, labels, phase)
                        running_loss += loss.data[0]
                        # backward + optimize only if in training phase
                else:
                    
                    labels = data['label'].type(torch.FloatTensor)
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())    
                    #print('with floattensor and cuda label={}'.format(labels))
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    outputs = model(inputs)
                    outputs = torch.mean(outputs)
                    #print('output after torch mean={} '.format(outputs))
                    loss = criterion(outputs, labels, phase)
                    running_loss += loss.data[0]
                        # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                #preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
                tempPreds = (outputs.data > 0.5).type(torch.FloatTensor)
                preds=Variable(torch.cuda.FloatTensor([tempPreds]), requires_grad=False)
                #preds=Variable(preds.cuda())
                running_corrects += torch.sum(preds == labels.data)
                #print('new Variable preds ={}, labels.data={} '.format(preds,labels.data))
                confusion_matrix[phase].add(preds, labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Confusion Meter:\n', confusion_matrix[phase].value())
            # deep copy the model
            torch.cuda.empty_cache()
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def predict_with_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    '''
    Loops over phase (train or valid) set to determine acc, loss and 
    confusion meter of the model.
    '''
    confusion_matrix = meter.ConfusionMeter(2, normalized=True)
    running_loss = 0.0
    running_corrects = 0
    norm_label_count =0
    abn_label_count=0
    abnormal_pred_count=0
    
    normal_pred_count =0
    for i, data in enumerate(dataloaders[phase]):
        with torch.no_grad():
            print(i, end='\r')
            #print("labels= {}".format(data['label']))
            if (data['label'] == 1.0):
                abn_label_count+=1
            if (data['label'] == 0.0):
                norm_label_count +=1
            labels = data['label'].type(torch.FloatTensor)
           
            #print("no of abnormal labels={} , normal labels={}".format(abn_label_count,norm_label_count))
            inputs = data['images'][0]
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            # forward
            outputs = model(inputs)
            outputs = torch.mean(outputs)
            loss = criterion(outputs, labels, phase)
            # statistics
            running_loss += loss.data[0] * inputs.size(0)
            #preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
            tempPreds = (outputs.data > 0.5).type(torch.FloatTensor)
            preds=Variable(torch.cuda.FloatTensor([tempPreds]), requires_grad=False)
            #preds=Variable(preds.cuda())
            #total corrects, abnormal or normal
            running_corrects += torch.sum(preds == labels.data)
            #print("variable preds={}".format(preds))
            if ((preds == 1.0) & (preds == labels.data)):
                abnormal_pred_count += 1
            if ((preds == 0.0) & (preds == labels.data)):
                normal_pred_count += 1
            #print("no of abnormal preds={}, no of normal preds={}".format(abnormal_pred_count,normal_pred_count))
            confusion_matrix.add(preds, labels.data)
    print("no of abnormal labels={} , normal labels={}".format(abn_label_count,norm_label_count))
    print("no of abnormal preds={}, no of normal preds={}".format(abnormal_pred_count,normal_pred_count))
    print("running_correct ={}".format(running_corrects))
    total_size=dataset_sizes[phase]
   
    loss = running_loss / dataset_sizes[phase]
    observed_acc = float(running_corrects)/ total_size
    print("observed_acc={}".format(observed_acc))
    abn_expected=(abn_label_count * abnormal_pred_count)/total_size
    norm_expected=(norm_label_count * normal_pred_count)/total_size
    expected_acc= (abn_expected+norm_expected)/total_size
    kappa=(observed_acc - expected_acc)/(1 - expected_acc)
    print('Loss: {:.4f},obsv_Acc: {},exp_Acc: {:.4f},kappa: {:.4f}'.format(loss, observed_acc,expected_acc,kappa))
    print('Confusion Meter:\n', confusion_matrix.value())
    
def plot_training(costs, accs):
    '''
    Plots curve of Cost vs epochs and Accuracy vs epochs for 'train' and 'valid' sets during training
    '''
    train_acc = accs['train']
    print('train_acc={}'.format(train_acc))
    valid_acc = accs['valid']
    print('valid_acc={}'.format(valid_acc))
    train_cost = costs['train']
    valid_cost = costs['valid']
    print('train_cost={}, valid_cost={}'.format(train_cost,valid_cost))
    epochs = range(len(train_acc))

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1,)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, valid_acc)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_cost)
    plt.plot(epochs, valid_cost)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Cost')
    
    plt.show()