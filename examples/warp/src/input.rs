//! The boundary with Bevy's input: stick and cursor vectors (glam) become gax directions here,
//! and nowhere else in the game touches glam (see `clippy.toml`; the functions that read
//! them are the only ones allowed to).

use crate::render::scene::HUD_HALF_HEIGHT;
use crate::sim;
use crate::sim::body::Pose;
use bevy::input::gamepad::{Gamepad, GamepadButton};
use bevy::prelude::{ButtonInput, KeyCode, MouseButton, Query, Window};
use gax::pga2d::Point;

/// Which device aimed last (the HUD's hints follow it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Device {
    /// Keyboard and mouse.
    #[default]
    Mouse,
    /// A gamepad.
    Pad,
}

/// Buttons for menus, pressed this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Menu {
    /// Start / confirm.
    pub start: bool,
    /// Pause / back.
    pub back: bool,
    /// Toggle the debug overlay.
    pub debug: bool,
    /// Toggle full screen.
    pub fullscreen: bool,
    /// Navigation: arrows, WASD, the D-pad, or a flick of the left stick.
    pub up: bool,
    /// Down.
    pub down: bool,
    /// Left.
    pub left: bool,
    /// Right.
    pub right: bool,
    /// A letter typed (for initials).
    pub letter: Option<char>,
    /// Backspace.
    pub erase: bool,
}

/// The Tunnel's extra controls.
#[derive(Clone, Copy, Debug, Default)]
pub struct Flight {
    /// The cursor in the HUD's units (`y` within `HUD_HALF_HEIGHT` of the centre), if it is
    /// over the window.
    pub cursor: Option<Point<(), f32>>,
    /// The right stick, when it is held: a direction.
    pub stick: Option<Point<(), f32>>,
    /// Barrel roll (edge): Q / E, or the bumpers.
    pub roll: i8,
    /// Brake to boost: Ctrl / Shift, or the triggers.
    pub throttle: f32,
    /// Bomb (edge): Space, the right mouse button, or Y.
    pub bomb: bool,
}

/// The left stick's menu direction, `-1`, `0` or `1` across and up; as a latch, the last
/// one over all pads, so that holding a stick moves once per flick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Latch(i8, i8);

impl Latch {
    /// This direction, or on an axis where it has none, `other`'s (the first pad pushed far
    /// enough flicks).
    fn or(self, other: Latch) -> Latch {
        let pick = |a: i8, b: i8| if a != 0 { a } else { b };
        Latch(pick(self.0, other.0), pick(self.1, other.1))
    }

    /// The menu moves where the sticks' combined direction `now` points if it differs from
    /// the last, once when it passes half way and again after it returns; `now` is kept.
    fn flick(&mut self, now: Latch, menu: &mut Menu) {
        if now.0 != 0 && now.0 != self.0 {
            menu.right |= now.0 > 0;
            menu.left |= now.0 < 0;
        }
        if now.1 != 0 && now.1 != self.1 {
            menu.up |= now.1 > 0;
            menu.down |= now.1 < 0;
        }
        *self = now;
    }
}

/// A deadzone with a smooth rescale: `|v| < dz` is zero, and the rest maps to `0..1`.
fn deadzone(x: f32, y: f32, dz: f32) -> Point<(), f32> {
    let v = Point::direction(x, y);
    let l = v.ideal_norm();
    if l < dz {
        return Point::zero();
    }
    v * (((l - dz) / (1.0 - dz)).min(1.0) / l)
}

/// Where the Plane's view is: its camera and half height, and the ship it aims from.
#[derive(Clone, Copy)]
pub struct View {
    pub camera: Pose,
    pub half_height: f32,
    pub ship: Point<(), f32>,
}

/// Read this frame's input: the simulation's (in gax types), the menu buttons, and the
/// Tunnel's controls. `device` follows what aimed last; `latch` keeps the menu flick.
pub fn read(
    keys: &ButtonInput<KeyCode>,
    mouse: &ButtonInput<MouseButton>,
    window: Option<&Window>,
    pads: &Query<&Gamepad>,
    view: View,
    device: &mut Device,
    latch: &mut Latch,
) -> (sim::Input, Menu, Flight) {
    let mut menu = menu_keys(keys);
    let mut flight = flight_keys(keys, mouse);
    let mut input = sim::Input {
        movement: movement_keys(keys),
        aim: Point::zero(),
        fire: mouse.pressed(MouseButton::Left),
        bomb: flight.bomb,
    };
    if let Some(w) = window {
        cursor(w, view, &mut input, &mut flight);
    }
    if input.fire || mouse.just_pressed(MouseButton::Left) {
        *device = Device::Mouse;
    }
    let mut flick = Latch::default();
    for pad in pads.iter() {
        if read_pad(pad, &mut input, &mut menu, &mut flight) {
            *device = Device::Pad;
        }
        flick = flick.or(stick_flick(pad));
    }
    latch.flick(flick, &mut menu);
    flight.throttle = flight.throttle.clamp(-1.0, 1.0);
    (input, menu, flight)
}

/// The keyboard's menu buttons.
fn menu_keys(keys: &ButtonInput<KeyCode>) -> Menu {
    const LETTERS: [KeyCode; 26] = [
        KeyCode::KeyA,
        KeyCode::KeyB,
        KeyCode::KeyC,
        KeyCode::KeyD,
        KeyCode::KeyE,
        KeyCode::KeyF,
        KeyCode::KeyG,
        KeyCode::KeyH,
        KeyCode::KeyI,
        KeyCode::KeyJ,
        KeyCode::KeyK,
        KeyCode::KeyL,
        KeyCode::KeyM,
        KeyCode::KeyN,
        KeyCode::KeyO,
        KeyCode::KeyP,
        KeyCode::KeyQ,
        KeyCode::KeyR,
        KeyCode::KeyS,
        KeyCode::KeyT,
        KeyCode::KeyU,
        KeyCode::KeyV,
        KeyCode::KeyW,
        KeyCode::KeyX,
        KeyCode::KeyY,
        KeyCode::KeyZ,
    ];
    let letter = LETTERS.iter().position(|k| keys.just_pressed(*k));
    Menu {
        start: keys.any_just_pressed([KeyCode::Enter, KeyCode::NumpadEnter]),
        back: keys.just_pressed(KeyCode::Escape),
        debug: keys.just_pressed(KeyCode::F3),
        fullscreen: keys.just_pressed(KeyCode::F11),
        up: keys.any_just_pressed([KeyCode::ArrowUp, KeyCode::KeyW]),
        down: keys.any_just_pressed([KeyCode::ArrowDown, KeyCode::KeyS]),
        left: keys.any_just_pressed([KeyCode::ArrowLeft, KeyCode::KeyA]),
        right: keys.any_just_pressed([KeyCode::ArrowRight, KeyCode::KeyD]),
        letter: letter.map(|i| (b'A' + i as u8) as char),
        erase: keys.just_pressed(KeyCode::Backspace),
    }
}

/// The keyboard's movement: WASD or the arrows, a diagonal no faster (at most unit length).
fn movement_keys(keys: &ButtonInput<KeyCode>) -> Point<(), f32> {
    let axis = |neg: [KeyCode; 2], pos: [KeyCode; 2]| {
        f32::from(u8::from(keys.any_pressed(pos))) - f32::from(u8::from(keys.any_pressed(neg)))
    };
    let movement = Point::direction(
        axis(
            [KeyCode::KeyA, KeyCode::ArrowLeft],
            [KeyCode::KeyD, KeyCode::ArrowRight],
        ),
        axis(
            [KeyCode::KeyS, KeyCode::ArrowDown],
            [KeyCode::KeyW, KeyCode::ArrowUp],
        ),
    );
    let l = movement.ideal_norm();
    if l > 1.0 {
        movement * (1.0 / l)
    } else {
        movement
    }
}

/// The Tunnel's keys and buttons: roll, throttle and the bomb (the Plane's too).
fn flight_keys(keys: &ButtonInput<KeyCode>, mouse: &ButtonInput<MouseButton>) -> Flight {
    let held = |k: [KeyCode; 2]| f32::from(u8::from(keys.any_pressed(k)));
    Flight {
        cursor: None,
        stick: None,
        roll: i8::from(keys.just_pressed(KeyCode::KeyE))
            - i8::from(keys.just_pressed(KeyCode::KeyQ)),
        throttle: held([KeyCode::ShiftLeft, KeyCode::ShiftRight])
            - held([KeyCode::ControlLeft, KeyCode::ControlRight]),
        bomb: keys.just_pressed(KeyCode::Space) || mouse.just_pressed(MouseButton::Right),
    }
}

/// The cursor: in the HUD's units for the Tunnel's reticle, and placed in the world through
/// the Plane's camera motor for its aim.
#[allow(clippy::disallowed_methods)] // the cursor's glam vector, read here only
fn cursor(w: &Window, view: View, input: &mut sim::Input, flight: &mut Flight) {
    let Some(c) = w.cursor_position() else {
        return;
    };
    let (width, height) = (w.width().max(1.0), w.height().max(1.0));
    let (nx, ny) = (2.0 * c.x / width - 1.0, 1.0 - 2.0 * c.y / height);
    let aspect = width / height;
    flight.cursor = Some(Point::xy(
        nx * HUD_HALF_HEIGHT * aspect,
        ny * HUD_HALF_HEIGHT,
    ));
    let local = Point::xy(nx * view.half_height * aspect, ny * view.half_height);
    input.aim = (view.camera >> local) - view.ship;
}

/// A gamepad: the left stick moves, the right stick aims and fires, the shoulders roll and
/// the triggers set the throttle (any of them bombs in the Plane, Y in the Tunnel), and the
/// face buttons and the D-pad work the menus. Returns whether a stick was pushed.
#[allow(clippy::disallowed_methods)] // the sticks' glam vectors, read here only
fn read_pad(pad: &Gamepad, input: &mut sim::Input, menu: &mut Menu, flight: &mut Flight) -> bool {
    let (ls, rs) = (pad.left_stick(), pad.right_stick());
    let left = deadzone(ls.x, ls.y, 0.15);
    let right = deadzone(rs.x, rs.y, 0.25);
    let moved = left.ideal_norm() > 0.0;
    let aimed = right.ideal_norm() > 0.0;
    if moved {
        input.movement = left;
    }
    if aimed {
        input.aim = right;
        input.fire = true;
        flight.stick = Some(right);
    }
    let pressed = |b: GamepadButton| pad.just_pressed(b);
    let held = |b: GamepadButton| f32::from(u8::from(pad.pressed(b)));
    flight.roll += i8::from(pressed(GamepadButton::RightTrigger))
        - i8::from(pressed(GamepadButton::LeftTrigger));
    flight.throttle += held(GamepadButton::RightTrigger2) - held(GamepadButton::LeftTrigger2);
    flight.bomb |= pressed(GamepadButton::North);
    let triggers = [
        GamepadButton::RightTrigger,
        GamepadButton::LeftTrigger,
        GamepadButton::RightTrigger2,
        GamepadButton::LeftTrigger2,
    ];
    input.bomb |= triggers.into_iter().any(pressed);
    menu.start |= pressed(GamepadButton::Start) || pressed(GamepadButton::South);
    menu.back |= pressed(GamepadButton::Select) || pressed(GamepadButton::East);
    menu.up |= pressed(GamepadButton::DPadUp);
    menu.down |= pressed(GamepadButton::DPadDown);
    menu.left |= pressed(GamepadButton::DPadLeft);
    menu.right |= pressed(GamepadButton::DPadRight);
    moved || aimed
}

/// The left stick's menu direction: `±1` on an axis pushed past half way, else 0.
#[allow(clippy::disallowed_methods)] // the stick's glam vector, read here only
fn stick_flick(pad: &Gamepad) -> Latch {
    let ls = pad.left_stick();
    let dir = |v: f32| {
        if v > 0.6 {
            1
        } else if v < -0.6 {
            -1
        } else {
            0
        }
    };
    Latch(dir(ls.x), dir(ls.y))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One pad held to the right and another at rest: the menu moves once, not every frame
    /// (each pad's latch used to overwrite the other's).
    #[test]
    fn a_held_stick_flicks_once_with_two_pads() {
        let (held, rest) = (Latch(1, 0), Latch(0, 0));
        let mut latch = Latch::default();
        let mut moves = 0;
        for _ in 0..5 {
            let mut menu = Menu::default();
            latch.flick(held.or(rest), &mut menu);
            moves += usize::from(menu.right);
            let mut menu = Menu::default();
            latch.flick(rest.or(held), &mut menu);
            moves += usize::from(menu.right);
        }
        assert_eq!(moves, 1);
        // Back to rest, and flicked again: another move, up this time from the other pad.
        let mut menu = Menu::default();
        latch.flick(rest.or(rest), &mut menu);
        latch.flick(rest.or(Latch(0, 1)), &mut menu);
        assert!(menu.up && !menu.right);
    }
}
