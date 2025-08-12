use super::Constraint;
use crate::core::OptFloat;
/// The whole space, no constraints
#[derive(Default, Clone, Copy)]
pub struct NoConstraints {}

impl NoConstraints {
    /// Constructs new instance of `NoConstraints`
    ///
    pub fn new() -> NoConstraints {
        NoConstraints {}
    }
}

impl<T> Constraint<T> for NoConstraints
where
    T: OptFloat,
{
    fn project(&self, _x: &mut [T]) {}

    fn is_convex(&self) -> bool {
        true
    }
}
