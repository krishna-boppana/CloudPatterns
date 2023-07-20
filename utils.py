import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data import augmentation_basic


def compare_train_val_test_distributions(train_set,val_set,test_set,metric):
    #computing multinomial distributions
    train_dist = multi_label_distribution(train_set)
    val_dist = multi_label_distribution(val_set)
    test_dist = multi_label_distribution(test_set)
    print('Train Distribution:{}'.format(train_dist))
    print('Validation Distribution:{}'.format(val_dist))
    print('Test Distribution:{}'.format(test_dist))

    #Assuming metric is symmetric
    print('Pairwise Distribution Comparison')
    print('(train,val):{}'.format(metric(train_dist,val_dist)))
    print('(val,test):{}'.format(metric(val_dist,test_dist)))
    print('(train,test):{}'.format(metric(train_dist,test_dist)))
    

def bhattacharyya_coefficient(distribution1,distribution2):
    pointwise = distribution1*distribution2 #point-wise multiplication
    sqrt = torch.sqrt(pointwise)
    coeff = torch.sum(sqrt)
    return coeff


def plot_epoch_log():
    epoch_log = torch.load('./logs/epoch_log')
    df = pd.DataFrame.from_dict(epoch_log, orient='index',columns=['train','validation'])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    sns.lineplot(data = df,dashes=False,
                palette=[(0.212395, 0.359683, 0.55171),(0.153364, 0.497, 0.557724)])
    plt.savefig('./plots/loss.png',bbox_inches='tight',dpi=300)
    

def label_distribution(dataset):
    loader = DataLoader(dataset)
    count = torch.zeros(4)
    for y in loader:
        count = count + y[0]
        if torch.sum(y[0]).item() == 0:
            no_labels = no_labels + 1
    label_dist = count / (len(dataset))
    return label_dist

class torch_binary_lists:
    'Produces all binary lists of length n'
    def __init__(self,n):
        self.length = 2**n
        self.combinations = []
        self.binary_lists(n)
        self.torch_dict = {}
        self.construct_torch_dict()

    def binary_lists(self,n):
        if n  == 1:
            self.combinations.append([0.0])
            self.combinations.append([1.0])
            return self.combinations

        length = len((self.binary_lists(n-1)))
        for x in range(length):
            template = self.combinations.pop(0)
            comb1 = [0.0] + template
            comb2 = [1.0] + template

            self.combinations.append(comb1)
            self.combinations.append(comb2)

        return self.combinations
    
    def construct_torch_dict(self):
        for x in range(self.length):
            self.torch_dict[str(torch.tensor(self.combinations[x]))] = x


def multi_label_distribution(dataset):
    num_labels = 4
    binary_lists = torch_binary_lists(num_labels)
    comb_dict = binary_lists.torch_dict
    loader = DataLoader(dataset)
    distribution = torch.zeros(2**num_labels)
    for y in loader:
        index = comb_dict[str(y[0])]
        distribution[index] = distribution[index] + 1
    return distribution / len(dataset)


def get_sample_batch(data,augmentation,batch_size):
    data.augmentation = augmentation
    loader = DataLoader(data,batch_size = batch_size)
    iterator = iter(loader)
    sample = next(iterator)
    data.augmentation = None
    return sample


def loss_implementation(logits,labels):
    batch_size, num_labels, = labels.shape
    param1 = torch.sigmoid(logits)
    param2 = 1 - torch.sigmoid(logits)
    term1 = -labels*torch.log(param1)
    term2 = -(1-labels)*torch.log(param2)
    values = term1 + term2
    double_sum = torch.sum(values)
    loss = (1/(batch_size*num_labels))*double_sum
    return loss


def test_lost_implementation(dataset,model,batch_size):
    imgs,labels = get_sample_batch(dataset,augmentation_basic,batch_size)
    with torch.no_grad():
        predictions = model(imgs)
        loss_1 = loss_implementation(predictions,labels)
        loss_2 =  torch.nn.BCEWithLogitsLoss()(predictions,labels)
        loss_3 = torch.nn.MultiLabelSoftMarginLoss()(predictions,labels)
    return (loss_1,loss_2,loss_3)

class model_structure:
    def __init__(self,model):
        self.layer_num = 1
        self.model = model
        self.num_param_tensors = None
        
    def print_layers(self):
        stack = list(self.model.children())
        counter = 1
        while stack != []:
            module = stack.pop(0)
            module_children_list = list(module.children())
            if len(module_children_list) == 0:
                print(counter, module,"parameter shapes = " + str([i.shape for i in module.parameters()]))
                counter = counter + 1
            else: 
                stack = module_children_list + stack
                
    def named_children_keys(self):
        return dict(self.model.named_children()).keys()

    def named_parameters_with_grad(self):
        return [(name,params.shape,params.requires_grad) for name,params in self.model.named_parameters()]
    
    def num_params_tensors(self):
        if self.num_param_tensors == None:
            self.num_param_tensors = len([x for x in self.model.parameters()])
        return self.num_param_tensors