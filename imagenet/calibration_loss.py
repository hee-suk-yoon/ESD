import torch
import ipdb
import os 
import torch.nn as nn
import numpy as np
import csv 
import math
import torch


def SBECE(num_bins, confidences, correct,T,device):
    N = len(confidences)
    m = num_bins

    bin_boundaries = torch.linspace(0, 1, num_bins + 1) 
    bin_space = ((bin_boundaries[1] - bin_boundaries[0])/2) 
    bin_center_temp = bin_boundaries + bin_space 
    bin_center = bin_center_temp[:-1] 
    r_hat = confidences.view(1,N)
    r_hat_matrix = r_hat.expand(m,N)
    epsilon_hat = bin_center.view(m,1) 
    epsilon_hat_matrix = epsilon_hat.expand(m,N).to(device)

    columnwise_softmax = torch.nn.Softmax(dim=0)
    g = -(r_hat_matrix-epsilon_hat_matrix)**2/T
    g_matrix = columnwise_softmax(g)

    correctness_vector = correct.float().view(1,N)
    correctness_matrix = correctness_vector.expand(m,N)
    A_matrix = correctness_matrix*g_matrix

    S = torch.sum(g_matrix, dim = 1)
    S_check = torch.eq(S, torch.zeros(m).to(device)).float() 
    A_hat = torch.div(torch.sum(A_matrix,dim = 1),S+S_check).view(m,1)
    A_hat_matrix = A_hat.expand(m,N)
    temp_matrix = ((A_hat_matrix - r_hat_matrix)**2)*g_matrix
    return (1/N* torch.sum(temp_matrix))**(1/2)


def MMCE_unweighted(device, confidence, correct, kernel_theta = 0.4):
    n = len(correct)
    A = confidence.view(1,n).expand(n,n)
    A_T = A.T
    kernel_matrix = torch.exp(-torch.abs(A-A_T)/kernel_theta)
    correct_matrix = correct.view(1,n).expand(n,n).float()
    val =  (A-correct_matrix)*(A.T-correct_matrix.T)*(kernel_matrix)
    
    MMCE_m = torch.sum(val/n**2)
    if MMCE_m < 0:
        ipdb.set_trace()
    return MMCE_m

def ESD(device, confidence1, correct):
    N1 = len(confidence1) #
    val = correct.float() - confidence1 # 
    val = val.view(1,N1) 
    mask = torch.ones(N1,N1) - torch.eye(N1)
    mask = mask.to(device)
    confidence1_matrix = confidence1.expand(N1,N1) #row copying
    temp = (confidence1.view(1,N1).T).expand(N1,N1)
    tri = torch.le(confidence1_matrix,temp).float() 
    val_matrix = val.expand(N1,N1)
    x_matrix = torch.mul(val_matrix,tri)*mask
    mean_row = torch.sum(x_matrix, dim = 1)/(N1-1) #gbar _i
    x_matrix_squared = torch.mul(x_matrix, x_matrix)
    var = 1/(N1-2) * torch.sum(x_matrix_squared,dim=1) - (N1-1)/(N1-2) * torch.mul(mean_row,mean_row)
    d_k_sq_vector = torch.mul(mean_row, mean_row) - var/(N1-1)
    reg_loss = torch.sum(d_k_sq_vector)/N1
    
    return reg_loss

def ECE_Loss(num_bins, predictions, confidences, correct):
    #ipdb.set_trace()
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_accuracy = [0]*num_bins
    bin_confidence = [0]*num_bins
    bin_num_sample = [0]*num_bins

    for idx in range(len(predictions)):
        #prediction = predictions[idx]
        confidence = confidences[idx]
        bin_idx = -1
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin_idx += 1 
            bin_lower = bin_lower.item()
            bin_upper = bin_upper.item()
            #if bin_lower <= confidence and confidence < bin_upper:
            if bin_lower < confidence and confidence <= bin_upper:
                bin_num_sample[bin_idx] += 1
                bin_accuracy[bin_idx] += correct[idx]
                bin_confidence[bin_idx] += confidences[idx]
    
    for idx in range(num_bins):
        if bin_num_sample[idx] != 0:
            bin_accuracy[idx] = bin_accuracy[idx]/bin_num_sample[idx]
            bin_confidence[idx] = bin_confidence[idx]/bin_num_sample[idx]

    ece_loss = 0.0
    for idx in range(num_bins):
        temp_abs = abs(bin_accuracy[idx]-bin_confidence[idx])
        ece_loss += (temp_abs*bin_num_sample[idx])/len(predictions)

    return ece_loss, bin_accuracy, bin_confidence, bin_num_sample
    
def log_wandb_imagenet(args, model, train_dataloader, cal_dataloader, val_dataloader, test_dataloader, device, T, w, b):
    total_correct = 0
    total_correct_platt = 0
    softmax_layer = torch.nn.Softmax(dim=1)
    predictions = []
    correct = []
    confidence = []

    predictions_platt = []
    correct_platt = []
    confidence_platt = []

    confidence_temperature = []

    model.eval()
    total_data = 0
    with torch.no_grad():
        for batch_id_, (image,label) in enumerate(test_dataloader):
            total_data += image.size(0)

            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            output = softmax_layer(logits)
            num_class = output.size(1)
            output_platt = softmax_layer(torch.bmm((torch.diag(w).unsqueeze(0)).expand(logits.size(0),num_class,num_class),logits.unsqueeze(2)).view(logits.size(0),-1) + b.unsqueeze(0).expand(logits.size(0),-1))
            output_temperature = softmax_layer(logits/T)
            # print(output)

            confidence_temp,prediction_temp = torch.max(output, dim = 1)
            confidence_temp_platt, prediction_temp_platt = torch.max(output_platt, dim = 1)
            confidence_temp_temperature , prediction_temp_temperature = torch.max(output_temperature, dim = 1)

            prediction_temp = prediction_temp.cpu()
            confidence_temp = confidence_temp.cpu()
            
            prediction_temp_platt = prediction_temp_platt.cpu()
            #prediction_temp_temperature = prediction_temp_temperature.cpu()

            label = label.cpu()

            predictions = predictions + prediction_temp.tolist()
            predictions_platt = predictions_platt + prediction_temp_platt.tolist()

            confidence = confidence + confidence_temp.tolist()
            confidence_platt = confidence_platt + confidence_temp_platt.tolist()
            confidence_temperature = confidence_temperature + confidence_temp_temperature.tolist()

            correct = correct + label.eq(prediction_temp).tolist()
            correct_platt = correct_platt + label.eq(prediction_temp_platt).tolist()
            total_correct += torch.sum(label.eq(prediction_temp)).item()
            total_correct_platt += torch.sum(label.eq(prediction_temp_platt)).item()
    test_accuracy = total_correct/total_data
    test_accuracy_platt = total_correct_platt/total_data
    ece_test = ECE_Loss(20,predictions,confidence, correct)
    ece_test_tempscale = ECE_Loss(20,predictions,confidence_temperature,correct)
    ece_test_platt = ECE_Loss(20, predictions_platt, confidence_platt, correct_platt)

    total_correct = 0
    softmax_layer = torch.nn.Softmax(dim=1)
    predictions = []
    correct = []
    confidence = []

    model.eval()
    total_data = 0
    with torch.no_grad():
        for batch_id_, (image,label) in enumerate(val_dataloader):
            total_data += image.size(0)
            image = image.to(device)
            label = label.to(device)
            logits = model(image)
            output = softmax_layer(logits)
            # print(output)

            confidence_temp,prediction_temp = torch.max(output, dim = 1)

            prediction_temp = prediction_temp.cpu()
            confidence_temp = confidence_temp.cpu()
            
            label = label.cpu()

            predictions = predictions + prediction_temp.tolist()

            confidence = confidence + confidence_temp.tolist()

            correct = correct + label.eq(prediction_temp).tolist()
            total_correct += torch.sum(label.eq(prediction_temp)).item()
    val_accuracy = total_correct/total_data
    ece_val = ECE_Loss(20,predictions,confidence, correct)            

    return test_accuracy, test_accuracy_platt, ece_test[0], ece_test_tempscale[0], ece_test_platt[0], val_accuracy, ece_val[0]




















