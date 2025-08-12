use super::Constraint;
use crate::core::OptFloat;

#[derive(Copy, Clone)]
/// A Euclidean ball, that is, a set given by $B_2^r = \\{x \in \mathbb{R}^n {}:{} \Vert{}x{}\Vert \leq r\\}$
/// or a Euclidean ball centered at a point $x_c$, that is, $B_2^{x_c, r} = \\{x \in \mathbb{R}^n {}:{} \Vert{}x-x_c{}\Vert \leq r\\}$
pub struct Ball2<'a, T>
where
    T: OptFloat,
{
    center: Option<&'a [T]>,
    radius: T,
}

impl<'a, T> Ball2<'a, T>
where
    T: OptFloat,
{
    /// Construct a new Euclidean ball with given center and radius
    /// If no `center` is given, then it is assumed to be in the origin
    pub fn new(center: Option<&'a [T]>, radius: T) -> Self {
        assert!(radius > T::zero());

        Ball2 { center, radius }
    }
}

impl<'a, T> Constraint<T> for Ball2<'a, T>
where
    T: OptFloat,
{
    fn project(&self, x: &mut [T]) {
        if let Some(center) = &self.center {
            let mut norm_difference = T::zero();
            x.iter().zip(center.iter()).for_each(|(a, b)| {
                let diff_ = *a - *b;
                norm_difference += diff_ * diff_;
            });

            norm_difference = norm_difference.sqrt();

            if norm_difference > self.radius {
                x.iter_mut().zip(center.iter()).for_each(|(x, c)| {
                    *x = *c + self.radius * (*x - *c) / norm_difference;
                });
            }
        } else {
            let norm_x = crate::matrix_operations::norm2(x);
            if norm_x > self.radius {
                let norm_over_radius = norm_x / self.radius;
                x.iter_mut().for_each(|x_| *x_ = *x_ / norm_over_radius);
            }
        }
    }

    fn is_convex(&self) -> bool {
        true
    }
}
