//! The boundary with Bevy's input: stick and cursor vectors (glam) become gax directions here,
//! and nowhere else in the game touches glam (see `clippy.toml`).

// The one module allowed to read Bevy's glam vectors; they become gax types right away.
#![allow(clippy::disallowed_types, clippy::disallowed_methods)]

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
    /// The cursor in the HUD's screen units (`y` in `-18..18`), if it is over the window.
    pub cursor: Option<[f32; 2]>,
    /// The right stick, when it is held.
    pub stick: Option<[f32; 2]>,
    /// Barrel roll (edge): Q / E, or the bumpers.
    pub roll: i8,
    /// Brake to boost: Ctrl / Shift, or the triggers.
    pub throttle: f32,
    /// Bomb (edge): Space, the right mouse button, or Y.
    pub bomb: bool,
}

/// The left stick's last menu direction, so that holding it moves once per flick.
#[derive(Clone, Copy, Debug, Default)]
pub struct Latch(i8, i8);

/// A deadzone with a smooth rescale: `|v| < dz` is zero, and the rest maps to `0..1`.
fn deadzone(x: f32, y: f32, dz: f32) -> (f32, f32) {
    let l = (x * x + y * y).sqrt();
    if l < dz {
        return (0.0, 0.0);
    }
    let k = ((l - dz) / (1.0 - dz)).min(1.0) / l;
    (x * k, y * k)
}

/// Read this frame's input: the simulation's (in gax types) and the menu buttons.
///
/// `camera` and `half_height` place the cursor in the world; `ship` is where the ship is.
#[allow(clippy::too_many_arguments)]
pub fn read(
    keys: &ButtonInput<KeyCode>,
    mouse: &ButtonInput<MouseButton>,
    window: Option<&Window>,
    pads: &Query<&Gamepad>,
    camera: Pose,
    half_height: f32,
    ship: [f32; 2],
    device: &mut Device,
    latch: &mut Latch,
) -> (sim::Input, Menu, Flight) {
    let mut input = sim::Input::default();
    let mut menu = Menu {
        start: keys.any_just_pressed([KeyCode::Enter, KeyCode::NumpadEnter]),
        back: keys.just_pressed(KeyCode::Escape),
        debug: keys.just_pressed(KeyCode::F3),
        fullscreen: keys.just_pressed(KeyCode::F11),
        up: keys.any_just_pressed([KeyCode::ArrowUp, KeyCode::KeyW]),
        down: keys.any_just_pressed([KeyCode::ArrowDown, KeyCode::KeyS]),
        left: keys.any_just_pressed([KeyCode::ArrowLeft, KeyCode::KeyA]),
        right: keys.any_just_pressed([KeyCode::ArrowRight, KeyCode::KeyD]),
        letter: None,
        erase: keys.just_pressed(KeyCode::Backspace),
    };
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
    if let Some(i) = LETTERS.iter().position(|k| keys.just_pressed(*k)) {
        menu.letter = Some((b'A' + i as u8) as char);
    }
    // Keyboard movement.
    let axis = |neg: [KeyCode; 2], pos: [KeyCode; 2]| {
        f32::from(u8::from(keys.any_pressed(pos))) - f32::from(u8::from(keys.any_pressed(neg)))
    };
    let (mut mx, mut my) = (
        axis(
            [KeyCode::KeyA, KeyCode::ArrowLeft],
            [KeyCode::KeyD, KeyCode::ArrowRight],
        ),
        axis(
            [KeyCode::KeyS, KeyCode::ArrowDown],
            [KeyCode::KeyW, KeyCode::ArrowUp],
        ),
    );
    let l = (mx * mx + my * my).sqrt();
    if l > 1.0 {
        mx /= l;
        my /= l;
    }
    let mut flight = Flight {
        cursor: None,
        stick: None,
        roll: i8::from(keys.just_pressed(KeyCode::KeyE))
            - i8::from(keys.just_pressed(KeyCode::KeyQ)),
        throttle: f32::from(u8::from(
            keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]),
        )) - f32::from(u8::from(
            keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]),
        )),
        bomb: keys.just_pressed(KeyCode::Space) || mouse.just_pressed(MouseButton::Right),
    };
    // Mouse aim: the cursor placed in the world through the camera motor.
    let mut aim = (0.0, 0.0);
    if let Some(w) = window
        && let Some(c) = w.cursor_position()
    {
        let (width, height) = (w.width().max(1.0), w.height().max(1.0));
        let (nx, ny) = (2.0 * c.x / width - 1.0, 1.0 - 2.0 * c.y / height);
        flight.cursor = Some([nx * 18.0 * width / height, ny * 18.0]);
        let local = Point::xy(nx * half_height * width / height, ny * half_height);
        let [wx, wy] = (camera >> local).to_euclidean();
        aim = (wx - ship[0], wy - ship[1]);
    }
    input.fire = mouse.pressed(MouseButton::Left);
    input.bomb = mouse.just_pressed(MouseButton::Right) || keys.just_pressed(KeyCode::Space);
    if input.fire || mouse.just_pressed(MouseButton::Left) {
        *device = Device::Mouse;
    }
    // Gamepads: left stick moves, right stick aims and fires.
    for pad in pads.iter() {
        let ls = pad.left_stick();
        let rs = pad.right_stick();
        let (lx, ly) = deadzone(ls.x, ls.y, 0.15);
        let (rx, ry) = deadzone(rs.x, rs.y, 0.25);
        if lx != 0.0 || ly != 0.0 {
            (mx, my) = (lx, ly);
            *device = Device::Pad;
        }
        if rx != 0.0 || ry != 0.0 {
            aim = (rx, ry);
            input.fire = true;
            flight.stick = Some([rx, ry]);
            *device = Device::Pad;
        }
        flight.roll += i8::from(pad.just_pressed(GamepadButton::RightTrigger))
            - i8::from(pad.just_pressed(GamepadButton::LeftTrigger));
        flight.throttle += f32::from(u8::from(pad.pressed(GamepadButton::RightTrigger2)))
            - f32::from(u8::from(pad.pressed(GamepadButton::LeftTrigger2)));
        flight.bomb |= pad.just_pressed(GamepadButton::North);
        let bomb = [
            GamepadButton::RightTrigger,
            GamepadButton::LeftTrigger,
            GamepadButton::RightTrigger2,
            GamepadButton::LeftTrigger2,
        ];
        input.bomb |= bomb.iter().any(|b| pad.just_pressed(*b));
        menu.start |=
            pad.just_pressed(GamepadButton::Start) || pad.just_pressed(GamepadButton::South);
        menu.back |=
            pad.just_pressed(GamepadButton::Select) || pad.just_pressed(GamepadButton::East);
        menu.up |= pad.just_pressed(GamepadButton::DPadUp);
        menu.down |= pad.just_pressed(GamepadButton::DPadDown);
        menu.left |= pad.just_pressed(GamepadButton::DPadLeft);
        menu.right |= pad.just_pressed(GamepadButton::DPadRight);
        // A flick of the left stick: once when it passes half way, again after it returns.
        let dir = |v: f32| {
            if v > 0.6 {
                1
            } else if v < -0.6 {
                -1
            } else {
                0
            }
        };
        let (fx, fy) = (dir(ls.x), dir(ls.y));
        if fx != 0 && fx != latch.0 {
            menu.right |= fx > 0;
            menu.left |= fx < 0;
        }
        if fy != 0 && fy != latch.1 {
            menu.up |= fy > 0;
            menu.down |= fy < 0;
        }
        *latch = Latch(fx, fy);
    }
    input.movement = Point::direction(mx, my);
    input.aim = Point::direction(aim.0, aim.1);
    flight.throttle = flight.throttle.clamp(-1.0, 1.0);
    (input, menu, flight)
}
