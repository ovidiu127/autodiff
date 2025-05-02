import numpy as np
from grad_fn import DotBackward0,PermuteBackward0,NegBackward0,AddBackward0,SubBackward0

class Tensor:
    var_cnt = 0

    grad_fn = {
        None:lambda nodes,out:None,
        'DotBackward0':DotBackward0,
        'PermuteBackward0':PermuteBackward0,
        'NegBackward0':NegBackward0,
        'AddBackward0':AddBackward0,
        'SubBackward0':SubBackward0
    }
    
    def __init__(
        self,
        data,
        name='tensor',
        dtype=np.float32,
        requires_grad=False
    ):
        self.data = np.array(data).astype(dtype)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data).astype(dtype) if self.requires_grad else None
        
        self.name = f'{name}_{Tensor.var_cnt}'
        Tensor.var_cnt += 1

        self.grad_fn = None
        
        self._prev = list()

    def __repr__(self):
        return f"Tensor({self.data},name={self.name},{f'grad_fn=<{self.grad_fn}>' if self.grad_fn is not None else f'requires_grad={self.requires_grad}'})"

    def dot(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other,name='dot_untracked',dtype=self.data.dtype,requires_grad=False)
        out = Tensor(
            self.data.dot(other.data),
            name=f'dot_<{self.name}>_<{other.name}>',
            dtype=self.data.dtype,
            requires_grad=(self.requires_grad or other.requires_grad) 
        )
            
        if out.requires_grad:
            out.grad_fn = 'DotBackward0'
            out._prev = [self,other]

        return out

    def __add__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other,name='add_untracked',dtype=self.data.dtype,requires_grad=False)
        out = Tensor(
            self.data + other.data,
            name=f'add_<{self.name}>_<{other.name}>',
            dtype=self.data.dtype,
            requires_grad=(self.requires_grad or other.requires_grad) 
        )
            
        if out.requires_grad:
            out.grad_fn = 'AddBackward0'
            out._prev = [self,other]

        return out

    def __radd__(self,other):
        return Tensor(other) + self

    def __sub__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other,name='sub_untracked',dtype=self.data.dtype,requires_grad=False)
        out = Tensor(
            self.data - other.data,
            name=f'sub_<{self.name}>_<{other.name}>',
            dtype=self.data.dtype,
            requires_grad=(self.requires_grad or other.requires_grad) 
        )
            
        if out.requires_grad:
            out.grad_fn = 'SubBackward0'
            out._prev = [self,other]

        return out

    def __rsub__(self,other):
        return Tensor(other) - self

    @property
    def T(self):
        out = Tensor(
            self.data.T,
            name=f'T_<{self.name}>',
            dtype=self.data.dtype,
            requires_grad=self.requires_grad
        )

        if out.requires_grad:
            out.grad_fn = 'PermuteBackward0'
            out._prev = [self]

        return out

    def __neg__(self):
        out = Tensor(
            -self.data,
            name=f'neg_<{self.name}>',
            dtype=self.data.dtype,
            requires_grad=self.requires_grad
        )

        if out.requires_grad:
            out.grad_fn = 'NegBackward0'
            out._prev = [self]

        return out

    def detach(self):
        return Tensor(
            self.data,
            name=f'{self.name}_detach',
            dtype=self.data.dtype,
            requires_grad=False
        )

    @property
    def shape(self):
        return self.data.shape
        
    def backward(self):
        if self.data.size != 1:
            raise Exception('Cannot backpropagate from multiple values')

        self.grad = np.ones_like(self.data)

        _visited = set()
        _topo_order = []
        def _topo(node):
            if node not in _visited:
                _visited.add(node)
                if not node.requires_grad:
                    return
                    
                _topo_order.append(node)
                for parent in node._prev:
                    _topo(parent)

        _topo(self)

        for node in _topo_order:
            Tensor.grad_fn[node.grad_fn](node._prev,node)