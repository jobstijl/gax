// The extension of a map on 8D vectors to the 4-vectors (70 × 70 minors of 24 terms) is over
// the generator's limit: it is not emitted (the bivectors' is), and the error says what to build
// instead. The kinds are named after the Hubbard ring's states: names no standard algebra uses,
// so that the message reads the same with every algebra enabled.
gax::algebra! {
    algebra r8 "Euclidean 8D, R(8,0): orbitals, and states of two and four electrons.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1, e7 = 1, e8 = 1;
    kind Orbital = [e1, e2, e3, e4, e5, e6, e7, e8];
    kind TwoElectrons = [e12, e13, e14, e15, e16, e17, e18, e23, e24, e25, e26, e27, e28, e34, e35,
        e36, e37, e38, e45, e46, e47, e48, e56, e57, e58, e67, e68, e78];
    kind FourElectrons = [e1234, e1235, e1236, e1237, e1238, e1245, e1246, e1247, e1248, e1256,
        e1257, e1258, e1267, e1268, e1278, e1345, e1346, e1347, e1348, e1356, e1357, e1358, e1367,
        e1368, e1378, e1456, e1457, e1458, e1467, e1468, e1478, e1567, e1568, e1578, e1678, e2345,
        e2346, e2347, e2348, e2356, e2357, e2358, e2367, e2368, e2378, e2456, e2457, e2458, e2467,
        e2468, e2478, e2567, e2568, e2578, e2678, e3456, e3457, e3458, e3467, e3468, e3478, e3567,
        e3568, e3578, e3678, e4567, e4568, e4578, e4678, e5678];
}

fn main() {
    let t = r8::Orbital::<(r8::Orbital,), f64>::zero();
    let _ = t.outermorphism::<r8::TwoElectrons>();
    let _ = t.outermorphism::<r8::FourElectrons>();
}
