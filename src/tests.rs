use std::num::NonZeroUsize;

use crate::constraints::*;
use crate::core::fbs::*;
use crate::core::*;

#[test]
fn t_access() {
    let radius = 0.2_f64;
    let box_constraints = Ball2::new(None, radius);
    let problem = Problem::new(
        &box_constraints,
        super::mocks::my_gradient,
        super::mocks::my_cost,
    );
    let gamma = 0.1;
    let tolerance = 1e-6;

    let mut fbs_cache = FBSCache::new(NonZeroUsize::new(2).unwrap(), gamma, tolerance);
    let mut u = [0.0; 2];
    let mut optimizer = FBSOptimizer::new(problem, &mut fbs_cache);

    let status = optimizer.solve(&mut u).unwrap();

    assert!(status.has_converged());
    assert!(status.norm_fpr() < tolerance);
    assert!((-0.14896 - u[0]).abs() < 1e-4);
    assert!((0.13346 - u[1]).abs() < 1e-4);
}

#[test]
fn t_access_f32() {
    let radius = 0.2f32;
    let box_constraints = Ball2::new(None, radius);
    let problem = Problem::new(
        &box_constraints,
        super::mocks::my_gradient,
        super::mocks::my_cost,
    );
    let gamma = 0.1f32;
    let tolerance = 1e-6f32;

    let mut fbs_cache = FBSCache::new(NonZeroUsize::new(2).unwrap(), gamma, tolerance);
    let mut u = [0.0f32; 2];
    let mut optimizer = FBSOptimizer::new(problem, &mut fbs_cache);

    let status = optimizer.solve(&mut u).unwrap();

    assert!(status.has_converged());
    assert!(status.norm_fpr() < tolerance);
    assert!((-0.14896f32 - u[0]).abs() < 1e-4f32);
    assert!((0.13346f32 - u[1]).abs() < 1e-4f32);
}
