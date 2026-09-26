//! Asteroids on plane-based geometric algebra.
//!
//! Every body is a PGA2D motor moved by twists (`exp` of bivectors), shapes are placed with
//! the batch sandwich kernels, and hits are tested with signs of `line ∧ point`. The game
//! logic is in `game.rs`; this file draws it with macroquad.
//!
//! Keys: arrows or A/D to turn, Up or W to thrust, Space to shoot, Enter to restart, Esc to
//! quit.

mod game;

use game::{Game, Input, WORLD};
use macroquad::prelude::*;

fn window() -> Conf {
    Conf {
        window_title: "gax asteroids".to_owned(),
        window_width: 1120,
        window_height: 700,
        high_dpi: true,
        ..Conf::default()
    }
}

/// World to screen: the world's center at the screen's center, y up, uniform scale.
struct View {
    scale: f32,
    cx: f32,
    cy: f32,
}

impl View {
    fn new() -> View {
        let scale = (screen_width() / WORLD[0]).min(screen_height() / WORLD[1]);
        View {
            scale,
            cx: screen_width() / 2.0,
            cy: screen_height() / 2.0,
        }
    }
    fn at(&self, p: [f32; 2]) -> Vec2 {
        vec2(self.cx + p[0] * self.scale, self.cy - p[1] * self.scale)
    }
}

fn polygon(view: &View, pts: &[gax::pga2d::Point], color: Color, width: f32) {
    for (a, b) in pts.iter().zip(pts.iter().cycle().skip(1)) {
        let (a, b) = (view.at(a.to_euclidean()), view.at(b.to_euclidean()));
        // Skip edges that wrap across the screen.
        if a.distance(b) < view.scale * WORLD[1] / 3.0 {
            draw_line(a.x, a.y, b.x, b.y, width, color);
        }
    }
}

#[macroquad::main(window)]
async fn main() {
    let seed = macroquad::miniquad::date::now().to_bits();
    let mut game = Game::new(seed);
    let mut placed = Vec::new();
    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if game.over && is_key_pressed(KeyCode::Enter) {
            game = Game::new(macroquad::miniquad::date::now().to_bits());
        }
        let input = Input {
            left: is_key_down(KeyCode::Left) || is_key_down(KeyCode::A),
            right: is_key_down(KeyCode::Right) || is_key_down(KeyCode::D),
            thrust: is_key_down(KeyCode::Up) || is_key_down(KeyCode::W),
            fire: is_key_down(KeyCode::Space),
        };
        // Small fixed steps keep the motion smooth when frames are slow.
        let dt = get_frame_time().min(0.05);
        let steps = (dt / 0.01).ceil().max(1.0);
        for _ in 0..steps as usize {
            game.update(dt / steps, input);
        }

        clear_background(Color::from_rgba(8, 10, 20, 255));
        let view = View::new();
        let border = view.at([-WORLD[0] / 2.0, WORLD[1] / 2.0]);
        draw_rectangle_lines(
            border.x,
            border.y,
            WORLD[0] * view.scale,
            WORLD[1] * view.scale,
            1.0,
            Color::from_rgba(40, 50, 80, 255),
        );
        let line = 1.5f32.max(view.scale * 0.25);

        for a in &game.asteroids {
            a.shape.placed(a.body.pose, &mut placed);
            polygon(&view, &placed, Color::from_rgba(200, 200, 215, 255), line);
        }
        for b in &game.bullets {
            let p = view.at(b.pos);
            draw_circle(p.x, p.y, line * 1.2, Color::from_rgba(255, 230, 120, 255));
        }
        for s in &game.sparks {
            let p = view.at(s.pos);
            let alpha = (s.life * 400.0).min(255.0) as u8;
            draw_circle(p.x, p.y, line, Color::from_rgba(255, 160, 60, alpha));
        }
        if !game.over {
            let blink = game.shield > 0.0 && (get_time() * 8.0) as i64 % 2 == 0;
            if !blink {
                game.ship_shape.placed(game.ship.pose, &mut placed);
                polygon(&view, &placed, Color::from_rgba(120, 220, 255, 255), line);
                if game.thrusting && (get_time() * 20.0) as i64 % 2 == 0 {
                    let flame = [
                        gax::pga2d::Point::xy(-0.8, -1.6),
                        gax::pga2d::Point::xy(0.0, -3.6),
                        gax::pga2d::Point::xy(0.8, -1.6),
                    ]
                    .map(|p| game.ship.pose >> p);
                    polygon(&view, &flame, Color::from_rgba(255, 140, 40, 255), line);
                }
            }
        }

        let size = 28.0;
        draw_text(
            format!(
                "SCORE {}   WAVE {}   SHIPS {}",
                game.score, game.wave, game.lives
            ),
            20.0,
            36.0,
            size,
            WHITE,
        );
        if game.over {
            let msg = "GAME OVER - press Enter";
            let m = measure_text(msg, None, 48, 1.0);
            draw_text(
                msg,
                (screen_width() - m.width) / 2.0,
                screen_height() / 2.0,
                48.0,
                WHITE,
            );
        }
        next_frame().await;
    }
}
