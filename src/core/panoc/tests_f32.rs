use crate::core::panoc::panoc_engine::PANOCEngine;
use crate::core::panoc::*;
use crate::core::*;
use crate::{mocks, FunctionCallResult};

const N_DIM: usize = 2;
#[test]
fn t_panoc_init() {
    let radius = 0.2_f32;
    let ball = constraints::Ball2::new(None, radius);
    let problem = Problem::new(&ball, mocks::my_gradient, mocks::my_cost);
    let mut panoc_cache = PANOCCache::new(N_DIM, 1e-6, 5);

    {
        let mut panoc_engine = PANOCEngine::new(problem, &mut panoc_cache);
        let mut u = [0.75, -1.4];
        panoc_engine.init(&mut u).unwrap();
        assert!(2.549_509_967_743_775 > panoc_engine.cache.lipschitz_constant);
        assert!(0.372_620_625_931_781 < panoc_engine.cache.gamma, "gamma");
        println!("----------- {} ", panoc_engine.cache.cost_value);
        unit_test_utils::assert_nearly_equal(
            6.34125,
            panoc_engine.cache.cost_value,
            1e-4,
            1e-10,
            "cost value",
        );
        unit_test_utils::assert_nearly_equal_array(
            &[0.35, -3.05],
            &panoc_engine.cache.gradient_u,
            1e-4,
            1e-10,
            "gradient at u",
        );
    }
    println!("cache = {:#?}", &panoc_cache);
}

fn print_panoc_engine<GradientType, ConstraintType, CostType, T>(
    panoc_engine: &PANOCEngine<GradientType, ConstraintType, CostType, T>,
) where
    GradientType: Fn(&[T], &mut [T]) -> FunctionCallResult,
    CostType: Fn(&[T], &mut T) -> FunctionCallResult,
    ConstraintType: constraints::Constraint<T>,
    T: OptFloat + std::fmt::Debug,
{
    println!("> fpr       = {:?}", &panoc_engine.cache.gamma_fpr);
    println!("> fpr       = {:.2?}", panoc_engine.cache.norm_gamma_fpr);
    println!(
        "> L         = {:.3?}",
        panoc_engine.cache.lipschitz_constant
    );
    println!("> gamma     = {:.10?}", panoc_engine.cache.gamma);
    println!("> tau       = {:.3?}", panoc_engine.cache.tau);
    println!("> lbfgs dir = {:.11?}", panoc_engine.cache.direction_lbfgs);
}

#[test]
fn t_test_panoc_basic() {
    let bounds = constraints::Ball2::new(None, 0.2_f32);
    let problem = Problem::new(&bounds, mocks::my_gradient, mocks::my_cost);
    let tolerance = 1e-9;
    let mut panoc_cache = PANOCCache::new(2, tolerance, 5);
    let mut panoc_engine = PANOCEngine::new(problem, &mut panoc_cache);

    let mut u = [0.0, 0.0];
    panoc_engine.init(&mut u).unwrap();
    panoc_engine.step(&mut u).unwrap();
    let fpr0 = panoc_engine.cache.norm_gamma_fpr;
    println!("fpr0 = {}", fpr0);

    for i in 1..=100 {
        println!("----------------------------------------------------");
        println!("> iter      = {}", i);
        print_panoc_engine(&panoc_engine);
        println!("> u         = {:.14?}", u);
        if panoc_engine.step(&mut u) != Ok(true) {
            break;
        }
    }
    println!("final |fpr| = {}", panoc_engine.cache.norm_gamma_fpr);
    assert!(panoc_engine.cache.norm_gamma_fpr <= tolerance);
    unit_test_utils::assert_nearly_equal_array(&u, &mocks::SOLUTION_A_F32, 1e-6, 1e-8, "");
}

#[test]
fn t_test_panoc_hard() {
    let radius: f32 = 0.05_f32;
    let bounds = constraints::Ball2::new(None, radius);
    let problem = Problem::new(
        &bounds,
        mocks::hard_quadratic_gradient,
        mocks::hard_quadratic_cost,
    );
    let n: usize = 3;
    let lbfgs_memory: usize = 10;
    let tolerance_fpr: f32 = 1e-12;
    let mut panoc_cache = PANOCCache::new(n, tolerance_fpr, lbfgs_memory);
    let mut panoc_engine = PANOCEngine::new(problem, &mut panoc_cache);

    let mut u = [-20.0, 10., 0.2];
    panoc_engine.init(&mut u).unwrap();

    println!("L     = {}", panoc_engine.cache.lipschitz_constant);
    println!("gamma = {}", panoc_engine.cache.gamma);
    println!("sigma = {}", panoc_engine.cache.sigma);

    let mut i = 1;
    println!("\n*** ITERATION   1");
    while panoc_engine.step(&mut u) == Ok(true) && i < 100 {
        i += 1;
        println!("+ u_plus               = {:?}", u);
        println!("\n*** ITERATION {:3}", i);
    }

    println!("\nsol = {:?}", u);
    assert!(panoc_engine.cache.norm_gamma_fpr <= tolerance_fpr);
    unit_test_utils::assert_nearly_equal_array(&u, &mocks::SOLUTION_HARD_F32, 1e-6, 1e-8, "");
}

#[test]
fn t_test_panoc_rosenbrock() {
    let tolerance = 1e-12_f32;
    let a_param = 1.0;
    let b_param = 100.0;
    let cost_gradient = |u: &[f32], grad: &mut [f32]| -> FunctionCallResult {
        mocks::rosenbrock_grad(a_param, b_param, u, grad);
        Ok(())
    };
    let cost_function = |u: &[f32], c: &mut f32| -> FunctionCallResult {
        *c = mocks::rosenbrock_cost(a_param, b_param, u);
        Ok(())
    };
    let bounds = constraints::Ball2::new(None, 1.0);
    let problem = Problem::new(&bounds, cost_gradient, cost_function);
    let mut panoc_cache = PANOCCache::new(2, tolerance, 2).with_cbfgs_parameters(2.0, 1e-6, 1e-12);
    let mut panoc_engine = PANOCEngine::new(problem, &mut panoc_cache);
    let mut u_solution = [-1.5, 0.9];
    panoc_engine.init(&mut u_solution).unwrap();
    let mut idx = 1;
    while panoc_engine.step(&mut u_solution) == Ok(true) && idx < 50 {
        idx += 1;
    }
    assert!(panoc_engine.cache.norm_gamma_fpr <= tolerance);
    println!("u = {:?}", u_solution);
}

#[test]
fn t_zero_gamma_l() {
    let tolerance = 1e-8_f32;
    let mut panoc_cache = PANOCCache::new(1, tolerance, 5);
    let u = &mut [1e6];

    // Define the cost function and its gradient.
    let df = |u: &[f32], grad: &mut [f32]| -> Result<(), SolverError> {
        grad[0] = u[0].signum();

        Ok(())
    };

    let f = |u: &[f32], c: &mut f32| -> Result<(), SolverError> {
        *c = u[0].abs();
        Ok(())
    };

    let bounds = constraints::NoConstraints::new();

    // Problem statement.
    let problem = Problem::new(&bounds, df, f);

    let mut panoc_engine = PANOCOptimizer::new(problem, &mut panoc_cache).with_max_iter(100);

    // Invoke the solver.
    let _status = panoc_engine.solve(u);
    println!("norm_gamma_fpr = {}", panoc_cache.norm_gamma_fpr);
    println!("u = {:?}", u);
    println!("iters = {}", panoc_cache.iteration);
    assert!(panoc_cache.norm_gamma_fpr <= tolerance);
}

#[test]
fn t_zero_gamma_huber() {
    let tolerance = 1e-8_f32;
    let mut panoc_cache = PANOCCache::new(1, tolerance, 10);
    let u = &mut [1e6];
    let huber_delta = 1e-6;

    // Define the cost function and its gradient.
    let df = |u: &[f32], grad: &mut [f32]| -> Result<(), SolverError> {
        let u_abs = u[0].abs();
        if u_abs >= huber_delta {
            grad[0] = huber_delta * u[0].signum();
        } else {
            grad[0] = u[0];
        }
        Ok(())
    };

    let huber_norm = |u: &[f32], y: &mut f32| -> Result<(), SolverError> {
        let u_abs = u[0].abs();
        if u_abs <= huber_delta {
            *y = 0.5 * u[0].powi(2);
        } else {
            *y = huber_delta * (u_abs - 0.5 * huber_delta);
        }
        Ok(())
    };

    let bounds = constraints::BallInf::new(None, 10000.);

    // Problem statement.
    let problem = Problem::new(&bounds, df, huber_norm);

    let mut panoc_engine = PANOCOptimizer::new(problem, &mut panoc_cache).with_max_iter(100);

    // Invoke the solver.
    let _status = panoc_engine.solve(u);
    println!("norm_gamma_fpr = {}", panoc_cache.norm_gamma_fpr);
    println!("u = {:?}", u);
    println!("iters = {}", panoc_cache.iteration);
    assert!(panoc_cache.norm_gamma_fpr <= tolerance);
}
