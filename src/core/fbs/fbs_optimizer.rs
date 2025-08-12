//! FBS Algorithm
//!

use std::time;

use crate::core::fbs::fbs_engine::FBSEngine;
use crate::core::fbs::FBSCache;
use crate::core::{AlgorithmEngine, ExitStatus, OptFloat, Optimizer, Problem, SolverStatus};
use crate::{constraints, matrix_operations, FunctionCallResult, SolverError};

const MAX_ITER: usize = 100_usize;

/// Optimiser using forward-backward splitting iterations (projected gradient)
///
/// Note that an `FBSOptimizer` holds a reference to an instance of `FBSEngine`
/// which needs to be created externally. A mutable reference to that `FBSEgnine`
/// is provided to the optimizer.
///
/// The `FBSEngine` is supposed to be updated whenever you need to solve
/// a different optimization problem.
///
///
pub struct FBSOptimizer<'a, GradientType, ConstraintType, CostType, T>
where
    GradientType: Fn(&[T], &mut [T]) -> FunctionCallResult,
    CostType: Fn(&[T], &mut T) -> FunctionCallResult,
    ConstraintType: constraints::Constraint<T>,
    T: OptFloat,
{
    fbs_engine: FBSEngine<'a, GradientType, ConstraintType, CostType, T>,
    max_iter: usize,
    max_duration: Option<time::Duration>,
}

impl<'a, GradientType, ConstraintType, CostType, T>
    FBSOptimizer<'a, GradientType, ConstraintType, CostType, T>
where
    GradientType: Fn(&[T], &mut [T]) -> FunctionCallResult,
    CostType: Fn(&[T], &mut T) -> FunctionCallResult,
    ConstraintType: constraints::Constraint<T>,
    T: OptFloat,
{
    /// Constructs a new instance of `FBSOptimizer`
    ///
    /// ## Arguments
    ///
    /// - `problem`: problem definition
    /// - `cache`: instance of `FBSCache`
    pub fn new(
        problem: Problem<'a, GradientType, ConstraintType, CostType, T>,
        cache: &'a mut FBSCache<T>,
    ) -> Self {
        FBSOptimizer {
            fbs_engine: FBSEngine::new(problem, cache),
            max_iter: MAX_ITER,
            max_duration: None,
        }
    }

    /// Sets the tolerance
    ///
    /// ## Panics
    ///
    /// The method panics if the specified tolerance is not positive
    pub fn with_tolerance(
        self,
        tolerance: T,
    ) -> FBSOptimizer<'a, GradientType, ConstraintType, CostType, T> {
        assert!(tolerance > T::zero());

        self.fbs_engine.cache.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of iterations
    pub fn with_max_iter(
        mut self,
        max_iter: usize,
    ) -> FBSOptimizer<'a, GradientType, ConstraintType, CostType, T> {
        self.max_iter = max_iter;
        self
    }

    /// Sets the maximum number of iterations
    pub fn with_max_duration(
        mut self,
        max_duration: time::Duration,
    ) -> FBSOptimizer<'a, GradientType, ConstraintType, CostType, T> {
        self.max_duration = Some(max_duration);
        self
    }
}

impl<'life, GradientType, ConstraintType, CostType, T> Optimizer<T>
    for FBSOptimizer<'life, GradientType, ConstraintType, CostType, T>
where
    GradientType: Fn(&[T], &mut [T]) -> FunctionCallResult + 'life,
    CostType: Fn(&[T], &mut T) -> FunctionCallResult + 'life,
    ConstraintType: constraints::Constraint<T> + 'life,
    T: OptFloat,
{
    fn solve(&mut self, u: &mut [T]) -> Result<SolverStatus<T>, SolverError> {
        let now = instant::Instant::now();

        // Initialize - propagate error upstream, if any
        self.fbs_engine.init(u)?;

        let mut num_iter: usize = 0;
        let mut step_flag = self.fbs_engine.step(u)?;

        if let Some(dur) = self.max_duration {
            while step_flag && num_iter < self.max_iter && dur <= now.elapsed() {
                num_iter += 1;
                step_flag = self.fbs_engine.step(u)?
            }
        } else {
            while step_flag && num_iter < self.max_iter {
                num_iter += 1;
                step_flag = self.fbs_engine.step(u)?
            }
        }

        // cost at the solution [propagate error upstream]
        let mut cost_value: T = T::zero();
        (self.fbs_engine.problem.cost)(u, &mut cost_value)?;

        if !matrix_operations::is_finite(u) || !cost_value.is_finite() {
            return Err(SolverError::NotFiniteComputation);
        }

        // export solution status
        Ok(SolverStatus::new(
            if num_iter < self.max_iter {
                ExitStatus::Converged
            } else {
                ExitStatus::NotConvergedIterations
            },
            num_iter,
            now.elapsed(),
            self.fbs_engine.cache.norm_fpr,
            cost_value,
        ))
    }
}
