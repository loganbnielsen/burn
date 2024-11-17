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
    use burn::{
        backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
        tensor::Tensor,
    };

    #[test]
    fn python_analog() {
        type Backend = Autodiff<Autodiff<Wgpu>>;
        let device = WgpuDevice::default();

        let x: f32 = 2.0;
        let y: f32 = 7.0;

        println!("\n\nINIT:\n\n");
        let expr_x: Tensor<Backend, 1> = Tensor::from_floats([x], &device).require_grad();
        let expr_y: Tensor<Backend, 1> = Tensor::from_floats([y], &device).require_grad();
        let expr_2: Tensor<Backend, 1> = Tensor::from_floats([2.0], &device);
        println!("expr_x: {:?}\n", expr_x);
        println!("expr_y: {:?}\n", expr_y);
        println!("expr_2: {:?}\n", expr_2);

        // f(x,y) = (x + y + 2) * (y * y)
        // f(11,7) = 539
        println!("\n\nconstruct final_expr\n\n");
        let final_expr =
            (expr_x.clone() + expr_y.clone() + expr_2.clone()) * (expr_y.clone() * expr_y.clone());
        println!("final_expr: {:?}\n", final_expr);

        println!("\n\nFIRST_BACKWARDS:\n\n");
        let grads = final_expr.backward();

        // f(x,y) = xy^2 + y^3 + 2y^2
        // del f / del x = y^2
        //               = 49 @ (11,7)
        // del f / del y = 2xy + 3y^2 + 4y
        //               = 28 + 147 + 28 @ (11,7)
        //               = 203
        let grad_x = expr_y.grad(&grads).unwrap();
        assert!(
            grad_x.clone().into_data()
                == Tensor::<Backend, 1>::from_floats([203.0], &device).into_data()
        );

        println!("\n\nSECOND_BACKWARDS:\n\n");
        let grads_grad_x = grad_x.backward();
        // del^2 f / del x^2 = 0
        let grad_xx = expr_x.grad(&grads_grad_x);
        println!("grad_xx: {:?}", grad_xx);
        // del^2 f / del y del x = 2y
        //                       = 2 * 7 @ (11,7)
        //                       = 14
        let grad_yx = expr_y.grad(&grads_grad_x);
        println!("grad_yx: {:?}", grad_yx);
        let grad_y = expr_y.grad(&grads).unwrap();
        // We'll uncomment this once the rest is working.
        // let grads_grad_y = grad_y.backward();
        // // del^2 f / del y^2 = 2x + 6y + 4
        // //                   = 4 + 42 + 4 @ (11,7)
        // //                   = 50
        // let grad_yy = expr_y.grad(&grads_grad_y);
        // println!("grad_yy: {:?}", grad_yy);
        // // del^2 f / del x del y = 2y
        // //                       = 2 * 7 @ (11,7)
        // //                       = 14
        // let grad_xy = expr_x.grad(&grads_grad_y);
        // println!("grad_xy: {:?}", grad_xy);
    }
}
