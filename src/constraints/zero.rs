use super::Constraint;
use crate::core::OptFloat;
#[derive(Clone, Copy, Default)]
/// Set Zero, $\\{0\\}$
pub struct Zero {}

impl Zero {
    /// Constructs new instance of `Zero`
    pub fn new() -> Self {
        Zero {}
    }
}

impl<T> Constraint<T> for Zero
where
    T: OptFloat,
{
    /// Computes the projection on $\\{0\\}$, that is, $\Pi_{\\{0\\}}(x) = 0$
    /// for all $x$
    fn project(&self, x: &mut [T]) {
        x.iter_mut().for_each(|xi| *xi = T::zero());
    }

    fn is_convex(&self) -> bool {
        true
    }
}
