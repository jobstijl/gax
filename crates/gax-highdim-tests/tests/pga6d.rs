//! The closed-form logarithm and the plain sandwiches in 7D (three invariant planes, the cubic of
//! 6D; docs/log6d.md), on an algebra declared with `algebra!`: plane-based PGA of 6D Euclidean
//! space, R(6,0,1): rotations and translations.

mod common;

gax::algebra! {
    algebra pga6d "Plane-based PGA of 6D Euclidean space, R(6,0,1): rotations and translations.";
    basis e0 = 0, e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1;
    kind Scalar = [1];
    versor Vector = [e0, e1, e2, e3, e4, e5, e6];
    kind Bivector = [e01, e02, e12, e03, e13, e23, e04, e14, e24, e34, e05, e15, e25, e35, e45, e06, e16, e26, e36, e46, e56];
    versor Even = [1, e01, e02, e12, e03, e13, e23, e04, e14, e24, e34, e05, e15, e25, e35, e45, e06, e16, e26, e36, e46, e56, e0123, e0124, e0134, e0234, e1234, e0125, e0135, e0235, e1235, e0145, e0245, e1245, e0345, e1345, e2345, e0126, e0136, e0236, e1236, e0146, e0246, e1246, e0346, e1346, e2346, e0156, e0256, e1256, e0356, e1356, e2356, e0456, e1456, e2456, e3456, e012345, e012346, e012356, e012456, e013456, e023456, e123456];
}

// One basis plane per invariant plane (position in `Bivector`, rotation or not).
checks!(pga6d, checks, [(2, true), (9, true), (10, false)]);
