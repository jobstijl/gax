//! Running an example: a window that plays the animation (space pauses, `r` restarts, escape
//! quits), or, headless, a GIF of one loop (`--gif PATH`) or a PNG of one moment
//! (`--png PATH --at SECONDS`). `--size WxH` overrides the size.

use crate::canvas::Canvas;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// An animation's settings.
#[derive(Clone, Copy, Debug)]
pub struct Anim {
    /// The window's title (and the example's name).
    pub title: &'static str,
    /// Canvas size in pixels.
    pub size: [u32; 2],
    /// The length of one loop in seconds (the animation's time runs modulo it).
    pub seconds: f32,
    /// Frames per second of a GIF.
    pub fps: f32,
    /// Window pixels per canvas pixel (2 for the ray-traced examples, which shade per pixel).
    pub scale: u32,
}

impl Anim {
    /// An animation of `seconds` per loop at 640 x 480, 25 frames per second in GIFs.
    pub const fn new(title: &'static str, seconds: f32) -> Anim {
        Anim {
            title,
            size: [640, 480],
            seconds,
            fps: 25.0,
            scale: 1,
        }
    }

    /// The same at another size.
    pub const fn size(self, width: u32, height: u32) -> Anim {
        Anim {
            size: [width, height],
            ..self
        }
    }

    /// The same with `k` window pixels per canvas pixel.
    pub const fn scale(self, k: u32) -> Anim {
        Anim { scale: k, ..self }
    }
}

/// The frame at time `t` (for tests and files).
pub fn frame(anim: &Anim, t: f32, draw: &mut impl FnMut(&mut Canvas, f32)) -> Canvas {
    let mut c = Canvas::new(anim.size[0] as usize, anim.size[1] as usize);
    draw(&mut c, t);
    c
}

/// For tests: one frame of `draw` at `t` (320 × 180) puts something on the canvas.
pub fn assert_draws(mut draw: impl FnMut(&mut Canvas, f32), t: f32) {
    let c = frame(&Anim::new("t", 1.0).size(320, 180), t, &mut draw);
    assert!(gax_light::luma(c.mean()) > 0.0);
}

/// Write one loop of the animation as a GIF.
pub fn save_gif(
    anim: &Anim,
    path: &str,
    draw: &mut impl FnMut(&mut Canvas, f32),
) -> std::io::Result<()> {
    let [w, h] = anim.size;
    let file = std::fs::File::create(path)?;
    let mut enc =
        gif::Encoder::new(file, w as u16, h as u16, &[]).map_err(std::io::Error::other)?;
    enc.set_repeat(gif::Repeat::Infinite)
        .map_err(std::io::Error::other)?;
    let n = (anim.seconds * anim.fps).round().max(1.0) as usize;
    for i in 0..n {
        let mut rgba = frame(anim, i as f32 / anim.fps, draw).to_rgba();
        let mut f = gif::Frame::from_rgba_speed(w as u16, h as u16, &mut rgba, 10);
        f.delay = (100.0 / anim.fps).round() as u16;
        enc.write_frame(&f).map_err(std::io::Error::other)?;
    }
    eprintln!("wrote {path} ({n} frames)");
    Ok(())
}

/// Write the frame at time `t` as a PNG.
pub fn save_png(
    anim: &Anim,
    path: &str,
    t: f32,
    draw: &mut impl FnMut(&mut Canvas, f32),
) -> std::io::Result<()> {
    let [w, h] = anim.size;
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(std::io::Error::other)?;
    writer
        .write_image_data(&frame(anim, t, draw).to_rgba())
        .map_err(std::io::Error::other)?;
    eprintln!("wrote {path}");
    Ok(())
}

/// Run the example: headless output when asked for on the command line, else a window.
pub fn run(mut anim: Anim, mut draw: impl FnMut(&mut Canvas, f32) + 'static) {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    if let Some(size) = arg("--size") {
        let (w, h) = size.split_once('x').expect("--size WxH");
        anim.size = [w.parse().expect("a width"), h.parse().expect("a height")];
    }
    if let Some(path) = arg("--gif") {
        save_gif(&anim, &path, &mut draw).expect("the GIF");
        return;
    }
    if let Some(path) = arg("--png") {
        let t = arg("--at").map_or(0.0, |t| t.parse().expect("--at SECONDS"));
        save_png(&anim, &path, t, &mut draw).expect("the PNG");
        return;
    }
    let event_loop = EventLoop::new().expect("an event loop");
    let mut app = App {
        anim,
        draw,
        window: None,
        surface: None,
        start: Instant::now(),
        paused: None,
    };
    event_loop.run_app(&mut app).expect("the window");
}

struct App<F> {
    anim: Anim,
    draw: F,
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    start: Instant,
    /// The time shown while paused.
    paused: Option<f32>,
}

impl<F> App<F> {
    fn time(&self) -> f32 {
        self.paused
            .unwrap_or_else(|| self.start.elapsed().as_secs_f32() % self.anim.seconds.max(1e-3))
    }
}

impl<F: FnMut(&mut Canvas, f32)> ApplicationHandler for App<F> {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let [w, h] = self.anim.size;
        let k = self.anim.scale.max(1);
        let attrs = Window::default_attributes()
            .with_title(format!("gax · {}", self.anim.title))
            .with_inner_size(winit::dpi::PhysicalSize::new(w * k, h * k));
        let window = Rc::new(el.create_window(attrs).expect("a window"));
        let context = softbuffer::Context::new(window.clone()).expect("a softbuffer context");
        self.surface = Some(softbuffer::Surface::new(&context, window.clone()).expect("a surface"));
        self.window = Some(window);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match logical_key {
                Key::Named(NamedKey::Escape) => el.exit(),
                Key::Named(NamedKey::Space) => {
                    self.paused = match self.paused {
                        Some(t) => {
                            self.start = Instant::now() - std::time::Duration::from_secs_f32(t);
                            None
                        }
                        None => Some(self.time()),
                    };
                }
                Key::Character(c) if c == "r" => {
                    self.start = Instant::now();
                    self.paused = self.paused.map(|_| 0.0);
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl<F: FnMut(&mut Canvas, f32)> App<F> {
    fn redraw(&mut self) {
        let t = self.time();
        let (Some(window), Some(surface)) = (&self.window, &mut self.surface) else {
            return;
        };
        let size = window.inner_size();
        let (Some(ww), Some(wh)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };
        let k = self.anim.scale.max(1) as usize;
        let (cw, ch) = (
            (size.width as usize).div_ceil(k),
            (size.height as usize).div_ceil(k),
        );
        let mut canvas = Canvas::new(cw, ch);
        (self.draw)(&mut canvas, t);
        let px = canvas.to_xrgb();
        surface.resize(ww, wh).expect("a resize");
        let mut buf = surface.buffer_mut().expect("a buffer");
        let width = size.width as usize;
        for (y, row) in buf.chunks_mut(width).enumerate() {
            let src = &px[(y / k) * cw..];
            for (x, p) in row.iter_mut().enumerate() {
                *p = src[x / k];
            }
        }
        buf.present().expect("a frame");
    }
}
