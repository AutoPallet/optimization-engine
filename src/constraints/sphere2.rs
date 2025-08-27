use super::Constraint;
use crate::core::OptFloat;
#[derive(Copy, Clone)]
/// A Euclidean sphere, that is, a set given by $S_2^r = \\{x \in \mathbb{R}^n {}:{} \Vert{}x{}\Vert = r\\}$
/// or a Euclidean sphere centered at a point $x_c$, that is, $S_2^{x_c, r} = \\{x \in \mathbb{R}^n {}:{} \Vert{}x-x_c{}\Vert = r\\}$
pub struct Sphere2<'a, T>
where
    T: OptFloat,
{
    center: Option<&'a [T]>,
    radius: T,
}

impl<'a, T> Sphere2<'a, T>
where
    T: OptFloat,
{
    /// Construct a new Euclidean sphere with given center and radius
    /// If no `center` is given, then it is assumed to be in the origin
    pub fn new(center: Option<&'a [T]>, radius: T) -> Self {
        assert!(radius > T::zero());
        Sphere2 { center, radius }
    }
}

impl<'a, T> Constraint<T> for Sphere2<'a, T>
where
    T: OptFloat,
{
    /// Projection onto the sphere, $S_{r, c}$ with radius $r$ and center $c$.
    /// If $x\neq c$, the projection is uniquely defined by
    ///
    /// $$
    /// P_{S_{r, c}}(x) = c + r\frac{x-c}{\Vert{}x-c\Vert_2},
    /// $$
    ///
    /// but for $x=c$, the projection is multi-valued. In particular, let
    /// $y = P_{S_{r, c}}(c)$. Then $y_1 = c_1 + r$ and $y_i = c_i$ for
    /// $i=2,\ldots, n$.
    ///
    /// ## Arguments
    ///
    /// - `x`: The given vector $x$ is updated with the projection on the set
    ///
    fn project(&self, x: &mut [T]) {
        let epsilon = T::from(1e-12).unwrap();
        if let Some(center) = &self.center {
            let norm_difference = crate::matrix_operations::norm2_squared_diff(x, center).sqrt();
            if norm_difference <= epsilon {
                x.copy_from_slice(center);
                x[0] += self.radius;
                return;
            }
            x.iter_mut().zip(center.iter()).for_each(|(x, c)| {
                *x = *c + self.radius * (*x - *c) / norm_difference;
            });
        } else {
            let norm_x = crate::matrix_operations::norm2(x);
            if norm_x <= epsilon {
                x[0] += self.radius;
                return;
            }
            let norm_over_radius = self.radius / norm_x;
            x.iter_mut().for_each(|x_| *x_ = *x_ * norm_over_radius);
        }
    }

    /// Returns false (the sphere is not a convex set)
    ///
    fn is_convex(&self) -> bool {
        false
    }
}
