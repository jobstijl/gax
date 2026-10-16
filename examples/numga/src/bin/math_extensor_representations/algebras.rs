//! The nondegenerate signatures through dimension six, each its own algebra (gax declares an
//! algebra's kinds at compile time): the scalars, the full multivector (added by the macro), and
//! the states, numga's choice of representatives (`spinor_layout`), where they are fewer than
//! the blades. The state kinds were read off numga's layout; the tests check that the port of the
//! layout chooses the same. Cl(0,0) has no directions: its scalars are all of it.

gax::algebra! {
    algebra cl00 "Cl(0,0), the real numbers.";
    basis;
    kind Scalar = [1];
}

gax::algebra! {
    algebra cl10 "Cl(1,0).";
    basis e1 = 1;
    kind Scalar = [1];
}

gax::algebra! {
    algebra cl01 "Cl(0,1).";
    basis e1 = -1;
    kind Scalar = [1];
}

gax::algebra! {
    algebra cl20 "Cl(2,0).";
    basis e1 = 1, e2 = 1;
    kind Scalar = [1];
    kind Spinor = [1, e2];
}

gax::algebra! {
    algebra cl11 "Cl(1,1).";
    basis e1 = 1, e2 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2];
}

gax::algebra! {
    algebra cl02 "Cl(0,2).";
    basis e1 = -1, e2 = -1;
    kind Scalar = [1];
}

gax::algebra! {
    algebra cl30 "Cl(3,0).";
    basis e1 = 1, e2 = 1, e3 = 1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e23];
}

gax::algebra! {
    algebra cl21 "Cl(2,1).";
    basis e1 = 1, e2 = 1, e3 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e23];
}

gax::algebra! {
    algebra cl12 "Cl(1,2).";
    basis e1 = 1, e2 = -1, e3 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e23];
}

gax::algebra! {
    algebra cl03 "Cl(0,3).";
    basis e1 = -1, e2 = -1, e3 = -1;
    kind Scalar = [1];
}

gax::algebra! {
    algebra cl40 "Cl(4,0).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e4, e23, e24, e34, e234];
}

gax::algebra! {
    algebra cl31 "Cl(3,1).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e23];
}

gax::algebra! {
    algebra cl22 "Cl(2,2).";
    basis e1 = 1, e2 = 1, e3 = -1, e4 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e4, e24];
}

gax::algebra! {
    algebra cl13 "Cl(1,3).";
    basis e1 = 1, e2 = -1, e3 = -1, e4 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e4, e23, e24, e34, e234];
}

gax::algebra! {
    algebra cl04 "Cl(0,4).";
    basis e1 = -1, e2 = -1, e3 = -1, e4 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e1, e2, e4, e12, e14, e24, e124];
}

gax::algebra! {
    algebra cl50 "Cl(5,0).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e4, e5, e23, e24, e34, e25, e35, e45, e234, e235, e245, e345, e2345];
}

gax::algebra! {
    algebra cl41 "Cl(4,1).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e4, e23, e24, e34, e234];
}

gax::algebra! {
    algebra cl32 "Cl(3,2).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = -1, e5 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e5, e23, e25, e35, e235];
}

gax::algebra! {
    algebra cl23 "Cl(2,3).";
    basis e1 = 1, e2 = 1, e3 = -1, e4 = -1, e5 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e4, e5, e24, e25, e45, e245];
}

gax::algebra! {
    algebra cl14 "Cl(1,4).";
    basis e1 = 1, e2 = -1, e3 = -1, e4 = -1, e5 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e4, e5, e23, e24, e34, e25, e35, e45, e234, e235, e245, e345, e2345];
}

gax::algebra! {
    algebra cl05 "Cl(0,5).";
    basis e1 = -1, e2 = -1, e3 = -1, e4 = -1, e5 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e1, e2, e4, e12, e14, e24, e124];
}

gax::algebra! {
    algebra cl60 "Cl(6,0).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e4, e6, e23, e24, e34, e26, e36, e46, e234, e236, e246, e346, e2346];
}

gax::algebra! {
    algebra cl51 "Cl(5,1).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e4, e6, e23, e24, e34, e26, e36, e46, e234, e236, e246, e346, e2346];
}

gax::algebra! {
    algebra cl42 "Cl(4,2).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = -1, e6 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e4, e23, e24, e34, e234];
}

gax::algebra! {
    algebra cl33 "Cl(3,3).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = -1, e5 = -1, e6 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e6, e23, e26, e36, e236];
}

gax::algebra! {
    algebra cl24 "Cl(2,4).";
    basis e1 = 1, e2 = 1, e3 = -1, e4 = -1, e5 = -1, e6 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e4, e5, e6, e24, e25, e45, e26, e46, e56, e245, e246, e256, e456, e2456];
}

gax::algebra! {
    algebra cl15 "Cl(1,5).";
    basis e1 = 1, e2 = -1, e3 = -1, e4 = -1, e5 = -1, e6 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e2, e3, e4, e6, e23, e24, e34, e26, e36, e46, e234, e236, e246, e346, e2346];
}

gax::algebra! {
    algebra cl06 "Cl(0,6).";
    basis e1 = -1, e2 = -1, e3 = -1, e4 = -1, e5 = -1, e6 = -1;
    kind Scalar = [1];
    kind Spinor = [1, e1, e2, e4, e12, e14, e24, e124];
}
