//! OptFloat trait that combines Float functionality with optimization constants (and a few traits)
use num::Float;

/// Trait that combines Float functionality with optimization-specific constants
/// This allows different float types to have different optimization parameters
pub trait OptFloat:
    Float
    + std::iter::Sum<Self>
    + num::FromPrimitive
    + num::ToPrimitive
    + std::fmt::Debug
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
{
    /// Minimum estimated Lipschitz constant (initial estimate)
    fn min_l_estimate() -> Self;

    /// gamma = GAMMA_L_COEFF/L
    fn gamma_l_coeff() -> Self;

    /// Delta in the estimation of the initial Lipschitz constant
    fn delta_lipschitz() -> Self;

    /// Epsilon in the estimation of the initial Lipschitz constant
    fn epsilon_lipschitz() -> Self;

    /// Safety parameter used to check a strict inequality in the update of the Lipschitz constant
    fn lipschitz_update_epsilon() -> Self;

    /// Maximum possible Lipschitz constant
    fn max_lipschitz_constant() -> Self;
}

/// Default implementation for f64 with original constants
impl OptFloat for f64 {
    fn min_l_estimate() -> Self {
        1e-10
    }

    fn gamma_l_coeff() -> Self {
        0.95
    }

    fn delta_lipschitz() -> Self {
        1e-12
    }

    fn epsilon_lipschitz() -> Self {
        1e-6
    }

    fn lipschitz_update_epsilon() -> Self {
        1e-6
    }

    fn max_lipschitz_constant() -> Self {
        1e9
    }
}

/// Default implementation for f32 with scaled constants
impl OptFloat for f32 {
    fn min_l_estimate() -> Self {
        8.74e-6
    }

    fn gamma_l_coeff() -> Self {
        0.95
    }

    fn delta_lipschitz() -> Self {
        1.32e-6
    }

    fn epsilon_lipschitz() -> Self {
        7.32e-4
    }

    fn lipschitz_update_epsilon() -> Self {
        2.62e-4
    }

    fn max_lipschitz_constant() -> Self {
        1e9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_constants() {
        assert_eq!(f64::min_l_estimate(), 1e-10);
        assert_eq!(f64::gamma_l_coeff(), 0.95);
        assert_eq!(f64::delta_lipschitz(), 1e-12);
        assert_eq!(f64::epsilon_lipschitz(), 1e-6);
        assert_eq!(f64::lipschitz_update_epsilon(), 1e-6);
        assert_eq!(f64::max_lipschitz_constant(), 1e9);
    }

    #[test]
    fn test_f32_constants() {
        assert_eq!(f32::min_l_estimate(), 8.74e-6);
        assert_eq!(f32::gamma_l_coeff(), 0.95);
        assert_eq!(f32::delta_lipschitz(), 1.32e-6);
        assert_eq!(f32::epsilon_lipschitz(), 7.32e-4);
        assert_eq!(f32::lipschitz_update_epsilon(), 2.62e-4);
        assert_eq!(f32::max_lipschitz_constant(), 1e9);
    }
}
