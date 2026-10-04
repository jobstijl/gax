//! A colour: a homogeneous point of its space, and the geometry of colour on it.

use crate::ops::{self, Raw};
use crate::space::{self, Space};
use core::marker::PhantomData;
use gax::pga3d::{Line, Motor, Plane, Point};

/// A colour of space `S`: a homogeneous point whose position is the colour and whose weight is
/// how much of it there is, its alpha (paint's coverage, or a light's intensity: premultiplied
/// RGBA is exactly a homogeneous point, so the two are one notion). Adding points composites:
/// `top + bottom * (1 - alpha(top))` is "over", and lights add as they are.
pub struct Colour<S> {
    point: Raw,
    space: PhantomData<S>,
}

impl<S> Clone for Colour<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S> Copy for Colour<S> {}
impl<S> PartialEq for Colour<S> {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
    }
}
impl<S: Space> core::fmt::Debug for Colour<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let p = self.point;
        write!(
            f,
            "{}({}, {}, {} | {})",
            S::NAME,
            p.e032(),
            p.e013(),
            p.e021(),
            p.e123()
        )
    }
}

/// A colour of linear sRGB: the renderer's colours, whose coordinates are radiance.
pub type LinearRgb = Colour<space::LinearRgb>;
/// A colour of sRGB as stored and shown.
pub type Srgb = Colour<space::Srgb>;
/// A colour of CIE XYZ.
pub type Xyz = Colour<space::Xyz>;
/// A colour of CIELAB.
pub type Lab = Colour<space::Lab>;
/// A colour of Oklab.
pub type Oklab = Colour<space::Oklab>;

impl<S: Space> Colour<S> {
    /// The colour of the point `point` of this space.
    pub const fn from_point(point: Raw) -> Self {
        Colour {
            point,
            space: PhantomData,
        }
    }

    /// The colour at `(x, y, z)` of this space with alpha `alpha`.
    pub const fn new(x: f32, y: f32, z: f32, alpha: f32) -> Self {
        Colour::from_point(Point::new(x * alpha, y * alpha, z * alpha, alpha))
    }

    /// The homogeneous point.
    pub const fn point(self) -> Raw {
        self.point
    }

    /// The point of weight 1 at the colour: the colour, fully there.
    pub fn unit(self) -> Raw {
        self.point.unitized()
    }

    /// No colour at all: weight 0.
    pub const fn transparent() -> Self {
        Colour::from_point(Point::new(0.0, 0.0, 0.0, 0.0))
    }

    /// The alpha: the weight.
    pub fn alpha(self) -> f32 {
        self.point.e123()
    }

    /// The same colour with alpha `alpha`.
    pub fn with_alpha(self, alpha: f32) -> Self {
        if self.alpha() == 0.0 {
            return Colour::from_point(S::black() * alpha);
        }
        Colour::from_point(self.point * (alpha / self.alpha()))
    }

    /// The same colour, `k` times as much of it.
    pub fn faded(self, k: f32) -> Self {
        Colour::from_point(ops::fade(self.point, k))
    }

    /// Whether there is none of it.
    pub fn is_transparent(self) -> bool {
        self.alpha() == 0.0
    }

    /// The colour as another colour type (`c.to::<Oklab>()`), with the same alpha.
    pub fn to<C: From<Self>>(self) -> C {
        C::from(self)
    }

    /// The colour in space `T`, with the same alpha: through linear sRGB.
    pub fn convert<T: Space>(self) -> Colour<T> {
        let w = self.alpha();
        if w == 0.0 {
            return Colour::transparent();
        }
        Colour::from_point(T::from_linear(S::to_linear(self.unit())) * w)
    }

    /// The colour changed by `f` in Oklab, where changes look even (a turn of hue that keeps
    /// lightness, a perceptual desaturation), and brought back to this space.
    pub fn perceptually(self, f: impl FnOnce(crate::Oklab) -> crate::Oklab) -> Self
    where
        Self: From<crate::Oklab>,
        crate::Oklab: From<Self>,
    {
        Self::from(f(crate::Oklab::from(self)))
    }

    /// The colour `t` of the way to `other`, in this space: an affine combination of the two
    /// positions, and of their alphas.
    pub fn mix(self, other: Self, t: f32) -> Self {
        if self.is_transparent() || other.is_transparent() {
            return Colour::from_point(ops::mix(self.point, other.point, t));
        }
        let (a, b) = (self.unit(), other.unit());
        let w = self.alpha() + (other.alpha() - self.alpha()) * t;
        Colour::from_point((a + (b - a) * t) * w)
    }

    /// The distance between two colours' positions in this space (in CIELAB the CIE76 colour
    /// difference, in Oklab ΔE_OK): the norm of their join.
    pub fn distance(self, other: Self) -> f32 {
        (self.unit() & other.unit()).norm()
    }

    // The cylindrical geometry: hue, chroma and lightness about the grey axis.

    /// The grey axis, black to white, at unit length.
    fn axis() -> Line {
        let l = S::black() & S::white();
        l * l.norm().recip()
    }

    /// The plane through black orthogonal to the grey axis: lightness 0.
    fn black_plane() -> Plane {
        Self::axis() | S::black()
    }

    /// The grey of the same lightness: the colour's foot on the grey axis, where the plane
    /// through it orthogonal to the axis meets the axis.
    pub fn grey(self) -> Self {
        let foot = ((Self::axis() | self.unit()) ^ Self::axis()).unitized();
        Colour::from_point(foot * self.alpha())
    }

    /// The lightness: how far along the grey axis, black 0 and white 1 (the ratio of the
    /// distances of the colour and of white from the plane of black).
    pub fn lightness(self) -> f32 {
        let plane = Self::black_plane();
        (plane & self.unit()).s() / (plane & S::white()).s()
    }

    /// The same hue and chroma at lightness `l`: the colour moved along the grey axis.
    pub fn with_lightness(self, l: f32) -> Self {
        let step = (S::white() - S::black()) * (l - self.lightness());
        Colour::from_point((self.unit() + step) * self.alpha())
    }

    /// Lighter by `amount` of lightness.
    pub fn lighter(self, amount: f32) -> Self {
        self.with_lightness(self.lightness() + amount)
    }

    /// Darker by `amount` of lightness.
    pub fn darker(self, amount: f32) -> Self {
        self.with_lightness(self.lightness() - amount)
    }

    /// The chroma: the distance from the grey axis (the norm of the join of the axis and the
    /// colour), in this space's units.
    pub fn chroma(self) -> f32 {
        (Self::axis() & self.unit()).norm()
    }

    /// The same hue and lightness at chroma `c`: the colour dilated about its grey. A grey has
    /// no hue to keep and stays grey.
    pub fn with_chroma(self, c: f32) -> Self {
        let now = self.chroma();
        if now == 0.0 {
            return self;
        }
        self.saturated(c / now)
    }

    /// Chroma times `k`: away from the grey axis (`k > 1`) or towards it.
    pub fn saturated(self, k: f32) -> Self {
        let (p, g) = (self.unit(), self.grey().unit());
        Colour::from_point((g + (p - g) * k) * self.alpha())
    }

    /// `t` of the way to the grey of the same lightness.
    pub fn desaturated(self, t: f32) -> Self {
        self.saturated(1.0 - t)
    }

    /// The hue, in radians from 0 to 2π: the angle about the grey axis from the half-plane of
    /// hue zero to the colour's. The two half-planes are the joins of the axis with a point of
    /// each; their product is the rotation by twice the angle between them, its logarithm the
    /// angle times the axis, signed by the way it turns. A grey has hue 0.
    pub fn hue(self) -> f32 {
        let axis = Self::axis();
        let unit = |p: Plane| p * p.norm().recip();
        let (zero, here) = (axis & S::hue_zero(), axis & self.unit());
        if here.norm() < 1e-9 {
            return 0.0;
        }
        let turn = (unit(here) * unit(zero)).normalized();
        let log: Line = turn.log();
        let angle = log.norm();
        let tau = core::f32::consts::TAU;
        if (log | axis).s() < 0.0 {
            tau - angle
        } else {
            angle
        }
    }

    /// The colour with its hue turned by `angle` radians: a rotation about the grey axis, which
    /// keeps lightness and chroma.
    pub fn rotate_hue(self, angle: f32) -> Self {
        let turn = Motor::rotation(Self::axis(), angle);
        Colour::from_point((turn >> self.unit()) * self.alpha())
    }

    /// The colour at hue `hue`, the same lightness and chroma.
    pub fn with_hue(self, hue: f32) -> Self {
        self.rotate_hue(hue - self.hue())
    }

    /// The opposite hue: half a turn about the grey axis.
    pub fn complement(self) -> Self {
        self.rotate_hue(core::f32::consts::PI)
    }

    /// The colour `t` of the way to `other` around the grey axis: the hue turned the shorter way
    /// round, lightness, chroma and alpha in proportion (the cylindrical mixing of HSL and OkLCh,
    /// as one motion about the axis). A grey takes the other's hue.
    pub fn mix_hue(self, other: Self, t: f32) -> Self {
        let tau = core::f32::consts::TAU;
        let (from, to) = (self.hue(), other.hue());
        // The shorter way round; none when either is grey (a grey has no hue to turn from or to).
        let greys = self.chroma() == 0.0 || other.chroma() == 0.0;
        let turn = if greys {
            0.0
        } else {
            (to - from + core::f32::consts::PI).rem_euclid(tau) - core::f32::consts::PI
        };
        let start = if self.chroma() == 0.0 {
            self.with_chroma_towards(other)
        } else {
            self
        };
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        start
            .rotate_hue(turn * t)
            .with_chroma(lerp(self.chroma(), other.chroma()))
            .with_lightness(lerp(self.lightness(), other.lightness()))
            .with_alpha(lerp(self.alpha(), other.alpha()))
    }

    /// A grey given the hue of `other`, at a vanishing chroma, so that chroma can grow from it.
    fn with_chroma_towards(self, other: Self) -> Self {
        let towards = (other.unit() - other.grey().unit()) * 1e-6;
        Colour::from_point((self.unit() + towards) * self.alpha())
    }

    /// The grey of lightness `l`.
    pub fn grey_of(l: f32) -> Self {
        Colour::from_point(S::black() + (S::white() - S::black()) * l)
    }
}
