use crate::graph::{
    node::{ForwardNode, ForwardNodeRef, ForwardNodeState},
    ops::InitRecordedOps,
};
use burn_tensor::backend::Backend;

#[derive(Debug, Clone)]
pub struct ADTensor<const D: usize, B: Backend> {
    pub node: ForwardNodeRef<B::TensorPrimitive<D>>,
}

impl<B: Backend, const D: usize> ADTensor<D, B> {
    pub fn from_tensor(tensor: B::TensorPrimitive<D>) -> Self {
        let state = ForwardNodeState::new(tensor);
        let ops = InitRecordedOps::new();
        let ops = Box::new(ops);
        let node = ForwardNode::from_root(state, ops);
        let node = std::sync::Arc::new(node);

        Self { node }
    }
}

impl<B: Backend, const D: usize> ADTensor<D, B> {
    pub fn tensor(&self) -> B::TensorPrimitive<D> {
        self.node.state.value()
    }

    pub fn tensor_ref(&self) -> &B::TensorPrimitive<D> {
        self.node.state.value_ref()
    }
}