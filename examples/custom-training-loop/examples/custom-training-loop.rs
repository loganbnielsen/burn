use burn::backend::{Autodiff, Wgpu};

fn main() {
    custom_training_loop::run::<Autodiff<Wgpu>>(Default::default());
}

mod test {
    use burn::{backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, tensor::Tensor};

    #[test]
    fn trivial() {
        type Backend = Autodiff<Wgpu>;
        let device = WgpuDevice::default();
        let x: Tensor<Backend, 1> = Tensor::from_data([5.0], &device).require_grad();
        let grads = x.backward();
        let maybe_x_grad = x.grad(&grads);
        if let Some(x_grad) = maybe_x_grad {
            println!("{}", x_grad);
        } else {
            print!("No grad.");
        }

    }

    #[test]
    fn nested_trivial() {
        type Backend = Autodiff<Autodiff<Wgpu>>;
        let device = WgpuDevice::default();
        let x: Tensor<Backend, 1> = Tensor::from_data([5.0], &device).require_grad();
        let grads = x.backward();
        let maybe_grad_x = x.grad(&grads);
        if let Some(grad_x) = maybe_grad_x {
            println!("grad_x: {}", grad_x);
            let grad_x = grad_x.require_grad();
            let grads_grad_x = grad_x.backward();
            let maybe_grad_xx = x.grad(&grads_grad_x);
            if let Some(grad_xx) = maybe_grad_xx {
                println!("grad_xx: {}", grad_xx);
            } else {
                println!("No grad_xx.");
            }
        } else {
            print!("No grad_x.");
        }
    }

    #[test]
    fn nested_x2() {
        println!("create backend");
        type Backend = Autodiff<Autodiff<Wgpu>>;
        println!("create device");
        let device = WgpuDevice::default();
        println!("create tensor");
        let x: Tensor<Backend, 1> = Tensor::from_data([5.0], &device).require_grad();
        println!("multiply");
        let x2 = x.clone() * x.clone();
        println!("backward");
        let grads = x2.backward();
        println!("get grad");
        let maybe_grad_x = x.grad(&grads);
        if let Some(grad_x) = maybe_grad_x {
            println!("grad_x: {}", grad_x);
            let grad_x = grad_x.require_grad();
            let grads_grad_x = grad_x.backward();
            let maybe_grad_xx = x.grad(&grads_grad_x);
            if let Some(grad_xx) = maybe_grad_xx {
                println!("grad_xx: {}", grad_xx);
            } else {
                println!("No grad_xx.");
            }
        } else {
            print!("No grad_x.");
        }
    }
}