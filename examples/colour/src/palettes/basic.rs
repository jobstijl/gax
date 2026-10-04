//! Named colors from the CSS1 specification, also known as
//! [basic colors](https://en.wikipedia.org/wiki/Web_colors#Basic_colors).
//! This is the same set of colors used in the
//! [VGA graphics standard](https://en.wikipedia.org/wiki/Video_Graphics_Array).

use crate::Srgb;

/// <div style="background-color: #00FFFF; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const AQUA: Srgb = Srgb::rgb(0.0, 1.0, 1.0);
/// <div style="background-color: #000000; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const BLACK: Srgb = Srgb::rgb(0.0, 0.0, 0.0);
/// <div style="background-color: #0000FF; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const BLUE: Srgb = Srgb::rgb(0.0, 0.0, 1.0);
/// <div style="background-color: #FF00FF; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const FUCHSIA: Srgb = Srgb::rgb(1.0, 0.0, 1.0);
/// <div style="background-color: #808080; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const GRAY: Srgb = Srgb::rgb(0.5019608, 0.5019608, 0.5019608);
/// <div style="background-color: #008000; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const GREEN: Srgb = Srgb::rgb(0.0, 0.5019608, 0.0);
/// <div style="background-color: #00FF00; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const LIME: Srgb = Srgb::rgb(0.0, 1.0, 0.0);
/// <div style="background-color: #800000; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const MAROON: Srgb = Srgb::rgb(0.5019608, 0.0, 0.0);
/// <div style="background-color: #000080; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const NAVY: Srgb = Srgb::rgb(0.0, 0.0, 0.5019608);
/// <div style="background-color: #808000; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const OLIVE: Srgb = Srgb::rgb(0.5019608, 0.5019608, 0.0);
/// <div style="background-color: #800080; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const PURPLE: Srgb = Srgb::rgb(0.5019608, 0.0, 0.5019608);
/// <div style="background-color: #FF0000; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const RED: Srgb = Srgb::rgb(1.0, 0.0, 0.0);
/// <div style="background-color: #C0C0C0; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const SILVER: Srgb = Srgb::rgb(0.7529412, 0.7529412, 0.7529412);
/// <div style="background-color: #008080; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const TEAL: Srgb = Srgb::rgb(0.0, 0.5019608, 0.5019608);
/// <div style="background-color: #FFFFFF; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const WHITE: Srgb = Srgb::rgb(1.0, 1.0, 1.0);
/// <div style="background-color: #FFFF00; width: 10px; padding: 10px; border: 1px solid;"></div>
pub const YELLOW: Srgb = Srgb::rgb(1.0, 1.0, 0.0);
