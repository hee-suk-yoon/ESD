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

    bin_boundaries = torch.linspace(0, 1, num_bins + 1) #
    bin_space = ((bin_boundaries[1] - bin_boundaries[0])/2) #
    bin_center_temp = bin_boundaries + bin_space #
    bin_center = bin_center_temp[:-1] #
    r_hat = confidences.view(1,N)
    r_hat_matrix = r_hat.expand(m,N)
    epsilon_hat = bin_center.view(m,1) #
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

























