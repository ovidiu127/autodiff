import numpy as np

def _unbroadcast(grad,shape):
    while len(grad.shape) > len(shape):
        grad = np.sum(grad,axis=0)
        
    for i,dim in enumerate(shape):
        if dim == 1:
            grad = np.sum(grad,axis=i,keepdims=True)
            
    return grad

def DotBackward0(nodes,out):
    if nodes[0].requires_grad:
        nodes[0].grad += out.grad.dot(nodes[1].data.T)
    
    if nodes[1].requires_grad:
        nodes[1].grad += nodes[0].data.T.dot(out.grad)

def PermuteBackward0(nodes,out):
    if nodes[0].requires_grad:
        nodes[0].grad += out.grad.T

def NegBackward0(nodes,out):
    if nodes[0].requires_grad:
        nodes[0].grad -= out.grad

def AddBackward0(nodes,out):
    if nodes[0].requires_grad:
        grad = _unbroadcast(out.grad,nodes[0].grad.shape)
        nodes[0].grad += grad
    
    if nodes[1].requires_grad:
        grad = _unbroadcast(out.grad,nodes[1].grad.shape)
        nodes[1].grad += grad

def SubBackward0(nodes,out):
    if nodes[0].requires_grad:
        grad = _unbroadcast(out.grad,nodes[0].grad.shape)
        nodes[0].grad += grad
    
    if nodes[1].requires_grad:
        grad = _unbroadcast(out.grad,nodes[1].grad.shape)
        nodes[1].grad -= grad