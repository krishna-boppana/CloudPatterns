import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data import augmentation_basic, resize_maintain_aspect_ratio


def batch_prob(model,batch):
    '''
    Function that computes the probability corresponding to each label, given a batch of predictions
    :param model: trained model
    :param model: batch of augmented images
    '''
    model.eval()
    with torch.no_grad():
        predictions_logits = model(batch)
        predictions = torch.sigmoid(predictions_logits)
        return predictions
    
def metrics_with_TTA(model,data):
    '''
    Function that reports exact accuracy (examples correct/total examples), 
    partial accuracy (# of matching elements between predictions and ground truth)/(total examples * number of labels)), and
    per-label accuracy (label1 acc, label 2 acc, label 3 acc, label 4 acc)

    :param model: trained model 
    :param data: test dataset 
    '''

    num_labels = 4
    loader = DataLoader(data,batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    metrics = {}

    total = len(data)

    exact_num_correct = 0
    partial_num_correct = 0
    per_label_correct = torch.zeros(num_labels)

    model.eval()
    with torch.no_grad():
        for raw_imgs,labels in loader:
            raw_imgs = raw_imgs.to(device)
            labels = labels.to(device)
            # For every batch of data, produce 4 augmented batches, 1 basic, 3 w/ geometric transformations
            basic_imgs = augmentation_basic(raw_imgs)
            augmented_imgs_1 = augmentation_horizontal_vertical(raw_imgs)
            augmented_imgs_2 = augmentation_horizontal(raw_imgs)
            augmented_imgs_3 = augmentation_vertical(raw_imgs)

            # Predicting label probabilities for each batch
            prob_basic = batch_prob(model,basic_imgs)
            prob_1 = batch_prob(model,augmented_imgs_1)
            prob_2 = batch_prob(model,augmented_imgs_2)
            prob_3 = batch_prob(model,augmented_imgs_3)

            sum = prob_basic + prob_1 + prob_2 + prob_3
            avg = sum / 4
            
            #Computing binary label predictions (threshold = .5)
            label_predictions = (avg > .5)

            #comparing with ground truth labels
            compare_with_ground_truth = (label_predictions == labels)

            batch_partial_correct = torch.sum(compare_with_ground_truth)
            batch_per_label_correct = torch.sum(compare_with_ground_truth,axis = 0)
            batch_exact_correct = torch.sum((torch.sum(compare_with_ground_truth,axis = 1) == num_labels))
    
            partial_num_correct = partial_num_correct + batch_partial_correct.item() #.item automatically sends to cpu
            per_label_correct = per_label_correct + batch_per_label_correct.cpu() #sending tensor to cpu
            exact_num_correct = exact_num_correct + batch_exact_correct.item()
        
        metrics['partial_accuracy'] = partial_num_correct / (total*num_labels) # (1 - hamming loss)
        metrics['per_label_accuracy'] = per_label_correct / total
        metrics['exact_accuracy'] = exact_num_correct / total

        torch.save(metrics,'./logs/metrics.pt')
        
    print(metrics)

#testing augmentation
augmentation_horizontal = transforms.Compose([
    resize_maintain_aspect_ratio(),
    transforms.RandomHorizontalFlip(p = 1.0),
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ]
)

#testing augmentation
augmentation_horizontal_vertical = transforms.Compose([
    resize_maintain_aspect_ratio(),
    transforms.RandomHorizontalFlip(p = 1.0),
    transforms.RandomVerticalFlip(p = 1.0),
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ]
)

#testing augmentation
augmentation_vertical = transforms.Compose([
    resize_maintain_aspect_ratio(),
    transforms.RandomVerticalFlip(p = 1.0),
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ]
)