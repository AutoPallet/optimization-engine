use super::{Constraint, Simplex};
use crate::core::OptFloat;

#[derive(Copy, Clone)]
/// A norm-1 ball, that is, a set given by $B_1^r = \\{x \in \mathbb{R}^n {}:{} \Vert{}x{}\Vert_1 \leq r\\}$
/// or a ball-1 centered at a point $x_c$, that is, $B_1^{x_c, r} = \\{x \in \mathbb{R}^n {}:{} \Vert{}x-x_c{}\Vert_1 \leq r\\}$
pub struct Ball1<'a, T>
where
    T: OptFloat,
{
    center: Option<&'a [T]>,
    radius: T,
    simplex: Simplex<T>,
}

impl<'a, T> Ball1<'a, T>
where
    T: OptFloat + PartialOrd + Copy,
{
    /// Construct a new ball-1 with given center and radius.
    /// If no `center` is given, then it is assumed to be in the origin
    pub fn new(center: Option<&'a [T]>, radius: T) -> Self {
        assert!(radius > T::zero());
        let simplex = Simplex::new(radius);
        Ball1 {
            center,
            radius,
            simplex,
        }
    }

    fn project_on_ball1_centered_at_origin(&self, x: &mut [T]) {
        if crate::matrix_operations::norm1(x) > self.radius {
            // u = |x| (copied)
            let mut u = vec![T::zero(); x.len()];
            u.iter_mut()
                .zip(x.iter())
                .for_each(|(ui, &xi)| *ui = xi.abs());
            // u = P_simplex(u)
            self.simplex.project(&mut u);
            x.iter_mut()
                .zip(u.iter())
                .for_each(|(xi, &ui)| *xi = xi.signum() * ui);
        }
    }
}

impl<'a, T> Constraint<T> for Ball1<'a, T>
where
    T: OptFloat + PartialOrd + Copy,
{
    fn project(&self, x: &mut [T]) {
        if let Some(center) = &self.center {
            x.iter_mut()
                .zip(center.iter())
                .for_each(|(xi, &ci)| *xi -= ci);
            self.project_on_ball1_centered_at_origin(x);
            x.iter_mut()
                .zip(center.iter())
                .for_each(|(xi, &ci)| *xi += ci);
        } else {
            self.project_on_ball1_centered_at_origin(x);
        }
    }

    fn is_convex(&self) -> bool {
        true
    }
}
