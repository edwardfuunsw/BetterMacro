//! Finds a captured image snippet inside a screenshot.
//!
//! Scores are normalised cross-correlation on luminance, the same measure as
//! OpenCV's TM_CCOEFF_NORMED that Sikuli-style tools use: 1.0 is a pixel-exact
//! match, and uniform brightness or contrast changes do not lower the score.
//! The search runs on a downscaled copy first, then refines the strongest few
//! candidates at full resolution.

use std::io::Cursor;

/// Coarse-search candidates that are refined at full resolution.
const CANDIDATES: usize = 4;
/// The coarse pass shrinks the template until its short side is about this many pixels.
const COARSE_TEMPLATE_SIDE: usize = 12;
const MAX_COARSE_FACTOR: usize = 8;

/// Grayscale image with luminance samples in 0.0..=1.0, row-major.
#[derive(Debug, Clone)]
pub struct Gray {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    /// Top-left corner of the match in the searched image's pixels.
    pub x: usize,
    pub y: usize,
    pub score: f32,
}

/// A region of the searched image, in pixels, where matches are ignored.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }
}

pub struct DecodedPng {
    pub image: Gray,
    /// Pixels per point from the PNG's density chunk (2 for Retina screenshots).
    pub scale: Option<f64>,
}

pub fn decode_png(bytes: &[u8]) -> Result<DecodedPng, String> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("Unable to read image: {e}"))?;
    let scale = reader.info().pixel_dims.and_then(|dims| {
        (dims.unit == png::Unit::Meter && dims.xppu > 0)
            // 72 dpi is one pixel per point.
            .then(|| (f64::from(dims.xppu) * 0.0254 / 72.0).round().max(1.0))
    });
    let size = reader
        .output_buffer_size()
        .ok_or("Image is too large to decode")?;
    let mut buffer = vec![0; size];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|e| format!("Unable to decode image: {e}"))?;
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err("Unsupported indexed PNG".into()),
    };
    let pixels = buffer[..info.buffer_size()]
        .chunks_exact(channels)
        .map(|px| {
            if channels < 3 {
                f32::from(px[0]) / 255.0
            } else {
                (0.299 * f32::from(px[0]) + 0.587 * f32::from(px[1]) + 0.114 * f32::from(px[2]))
                    / 255.0
            }
        })
        .collect();
    Ok(DecodedPng {
        image: Gray {
            width: info.width as usize,
            height: info.height as usize,
            pixels,
        },
        scale,
    })
}

impl Gray {
    /// Averages `factor`×`factor` blocks; trailing partial blocks are dropped.
    pub fn downscaled(&self, factor: usize) -> Gray {
        if factor <= 1 {
            return self.clone();
        }
        let width = self.width / factor;
        let height = self.height / factor;
        let area = (factor * factor) as f32;
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for row in 0..factor {
                    let start = (y * factor + row) * self.width + x * factor;
                    sum += self.pixels[start..start + factor].iter().sum::<f32>();
                }
                pixels.push(sum / area);
            }
        }
        Gray {
            width,
            height,
            pixels,
        }
    }

    /// Bilinear resize, used when a snippet was captured at a different display scale.
    pub fn resized(&self, width: usize, height: usize) -> Gray {
        let width = width.max(1);
        let height = height.max(1);
        let x_ratio = self.width as f32 / width as f32;
        let y_ratio = self.height as f32 / height as f32;
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            let source_y = ((y as f32 + 0.5) * y_ratio - 0.5).clamp(0.0, (self.height - 1) as f32);
            let top = source_y.floor() as usize;
            let bottom = (top + 1).min(self.height - 1);
            let fy = source_y - top as f32;
            for x in 0..width {
                let source_x =
                    ((x as f32 + 0.5) * x_ratio - 0.5).clamp(0.0, (self.width - 1) as f32);
                let left = source_x.floor() as usize;
                let right = (left + 1).min(self.width - 1);
                let fx = source_x - left as f32;
                let at = |px: usize, py: usize| self.pixels[py * self.width + px];
                let upper = at(left, top) * (1.0 - fx) + at(right, top) * fx;
                let lower = at(left, bottom) * (1.0 - fx) + at(right, bottom) * fx;
                pixels.push(upper * (1.0 - fy) + lower * fy);
            }
        }
        Gray {
            width,
            height,
            pixels,
        }
    }
}

/// A template with its mean removed, ready for correlation.
struct Prepared {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
    norm: f64,
}

fn prepare(template: &Gray) -> Option<Prepared> {
    let count = template.pixels.len();
    if count == 0 {
        return None;
    }
    let mean = template.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / count as f64;
    let pixels: Vec<f32> = template
        .pixels
        .iter()
        .map(|&p| (f64::from(p) - mean) as f32)
        .collect();
    let norm = pixels.iter().map(|&p| f64::from(p).powi(2)).sum::<f64>().sqrt();
    // A near-uniform snippet correlates with noise anywhere on screen.
    (norm / (count as f64).sqrt() >= 0.01).then_some(Prepared {
        width: template.width,
        height: template.height,
        pixels,
        norm,
    })
}

/// Whether a snippet has enough contrast to be found reliably.
pub fn has_detail(template: &Gray) -> bool {
    prepare(template).is_some()
}

/// Cross term, sum and sum of squares over one row, in eight lanes so it vectorises.
fn row_sums(screen: &[f32], template: &[f32]) -> (f32, f32, f32) {
    let mut cross = [0.0f32; 8];
    let mut sum = [0.0f32; 8];
    let mut squares = [0.0f32; 8];
    let screen_chunks = screen.chunks_exact(8);
    let template_chunks = template.chunks_exact(8);
    let (screen_tail, template_tail) = (screen_chunks.remainder(), template_chunks.remainder());
    for (s, t) in screen_chunks.zip(template_chunks) {
        for lane in 0..8 {
            cross[lane] += s[lane] * t[lane];
            sum[lane] += s[lane];
            squares[lane] += s[lane] * s[lane];
        }
    }
    let mut totals = (
        cross.iter().sum::<f32>(),
        sum.iter().sum::<f32>(),
        squares.iter().sum::<f32>(),
    );
    for (&s, &t) in screen_tail.iter().zip(template_tail) {
        totals.0 += s * t;
        totals.1 += s;
        totals.2 += s * s;
    }
    totals
}

fn score_at(screen: &Gray, template: &Prepared, x: usize, y: usize) -> f32 {
    let count = (template.width * template.height) as f64;
    let (mut cross, mut sum, mut squares) = (0.0f64, 0.0f64, 0.0f64);
    for row in 0..template.height {
        let start = (y + row) * screen.width + x;
        let (c, s, q) = row_sums(
            &screen.pixels[start..start + template.width],
            &template.pixels[row * template.width..(row + 1) * template.width],
        );
        cross += f64::from(c);
        sum += f64::from(s);
        squares += f64::from(q);
    }
    let variance = squares - sum * sum / count;
    // Flat screen areas cannot match a template that has detail.
    if variance <= count * 1e-5 {
        return 0.0;
    }
    (cross / (template.norm * variance.sqrt())) as f32
}

/// Scores every placement of `template` in `screen`, split across threads by row.
fn score_map(screen: &Gray, template: &Prepared) -> (Vec<f32>, usize) {
    let columns = screen.width - template.width + 1;
    let rows = screen.height - template.height + 1;
    let mut scores = vec![0.0f32; columns * rows];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let rows_per_thread = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in scores.chunks_mut(rows_per_thread * columns).enumerate() {
            scope.spawn(move || {
                for (offset, score) in chunk.iter_mut().enumerate() {
                    let y = chunk_index * rows_per_thread + offset / columns;
                    *score = score_at(screen, template, offset % columns, y);
                }
            });
        }
    });
    (scores, columns)
}

/// Returns the best placement of `template` in `screen` whose centre is outside
/// `excluded`, or `None` when the template cannot be searched at all.
pub fn find(screen: &Gray, template: &Gray, excluded: &[Rect]) -> Option<Match> {
    if template.width > screen.width || template.height > screen.height {
        return None;
    }
    let full = prepare(template)?;
    let factor = (template.width.min(template.height) / COARSE_TEMPLATE_SIDE)
        .clamp(1, MAX_COARSE_FACTOR);
    // Downscaling can flatten a thin snippet; search those at full resolution.
    let (factor, coarse) = match prepare(&template.downscaled(factor)) {
        Some(coarse) if factor > 1 => (factor, coarse),
        _ => (1, prepare(template)?),
    };
    let downscaled;
    let coarse_screen = if factor > 1 {
        downscaled = screen.downscaled(factor);
        &downscaled
    } else {
        screen
    };
    let (mut scores, columns) = score_map(coarse_screen, &coarse);
    let rows = scores.len() / columns;
    let is_excluded = |x: usize, y: usize| {
        let centre_x = (x * factor) as f64 + template.width as f64 / 2.0;
        let centre_y = (y * factor) as f64 + template.height as f64 / 2.0;
        excluded.iter().any(|rect| rect.contains(centre_x, centre_y))
    };
    for (index, score) in scores.iter_mut().enumerate() {
        if is_excluded(index % columns, index / columns) {
            *score = f32::NEG_INFINITY;
        }
    }

    let mut candidates = Vec::new();
    for _ in 0..CANDIDATES {
        let Some((index, &best)) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
        else {
            break;
        };
        if !best.is_finite() {
            break;
        }
        let (x, y) = (index % columns, index / columns);
        candidates.push((x, y));
        // Suppress the rest of this peak so the next candidate is somewhere else.
        for sy in y.saturating_sub(coarse.height / 2)..(y + coarse.height / 2 + 1).min(rows) {
            for sx in x.saturating_sub(coarse.width / 2)..(x + coarse.width / 2 + 1).min(columns) {
                scores[sy * columns + sx] = f32::NEG_INFINITY;
            }
        }
    }

    let max_x = screen.width - template.width;
    let max_y = screen.height - template.height;
    let refined: Vec<Option<Match>> = std::thread::scope(|scope| {
        let handles: Vec<_> = candidates
            .iter()
            .map(|&(cx, cy)| {
                let full = &full;
                scope.spawn(move || {
                    let mut best: Option<Match> = None;
                    let (x0, y0) = (cx * factor, cy * factor);
                    for y in y0.saturating_sub(factor)..=(y0 + factor).min(max_y) {
                        for x in x0.saturating_sub(factor)..=(x0 + factor).min(max_x) {
                            let centre_x = x as f64 + template.width as f64 / 2.0;
                            let centre_y = y as f64 + template.height as f64 / 2.0;
                            if excluded.iter().any(|rect| rect.contains(centre_x, centre_y)) {
                                continue;
                            }
                            let score = score_at(screen, full, x, y);
                            if best.is_none_or(|b| score > b.score) {
                                best = Some(Match { x, y, score });
                            }
                        }
                    }
                    best
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().ok().flatten()).collect()
    });
    refined
        .into_iter()
        .flatten()
        .max_by(|a, b| a.score.total_cmp(&b.score))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic texture so tests do not depend on a real screen.
    fn noise(width: usize, height: usize, seed: u32) -> Gray {
        let mut state = seed;
        let pixels = (0..width * height)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        Gray {
            width,
            height,
            pixels,
        }
    }

    /// Smooth texture, closer to real UI than per-pixel noise.
    fn blurred_noise(width: usize, height: usize, seed: u32) -> Gray {
        let rough = noise(width + 4, height + 4, seed);
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for dy in 0..5 {
                    for dx in 0..5 {
                        sum += rough.pixels[(y + dy) * rough.width + x + dx];
                    }
                }
                pixels.push(sum / 25.0);
            }
        }
        Gray {
            width,
            height,
            pixels,
        }
    }

    fn crop(image: &Gray, x: usize, y: usize, width: usize, height: usize) -> Gray {
        let mut pixels = Vec::with_capacity(width * height);
        for row in y..y + height {
            pixels.extend_from_slice(&image.pixels[row * image.width + x..][..width]);
        }
        Gray {
            width,
            height,
            pixels,
        }
    }

    fn paste(image: &mut Gray, patch: &Gray, x: usize, y: usize) {
        for row in 0..patch.height {
            let start = (y + row) * image.width + x;
            image.pixels[start..start + patch.width]
                .copy_from_slice(&patch.pixels[row * patch.width..][..patch.width]);
        }
    }

    #[test]
    fn finds_an_exact_snippet_with_a_coarse_pass() {
        let screen = blurred_noise(640, 400, 7);
        let template = crop(&screen, 311, 187, 90, 40);
        let found = find(&screen, &template, &[]).unwrap();
        assert_eq!((found.x, found.y), (311, 187));
        assert!(found.score > 0.99);
    }

    #[test]
    fn small_snippets_are_searched_at_full_resolution() {
        let screen = noise(300, 200, 3);
        let template = crop(&screen, 42, 150, 14, 10);
        let found = find(&screen, &template, &[]).unwrap();
        assert_eq!((found.x, found.y), (42, 150));
    }

    #[test]
    fn brightness_and_contrast_changes_still_match() {
        let screen = blurred_noise(400, 300, 11);
        let mut template = crop(&screen, 120, 80, 60, 36);
        for pixel in &mut template.pixels {
            *pixel = *pixel * 0.7 + 0.2;
        }
        let found = find(&screen, &template, &[]).unwrap();
        assert_eq!((found.x, found.y), (120, 80));
        assert!(found.score > 0.99);
    }

    #[test]
    fn excluded_regions_are_skipped() {
        let mut screen = blurred_noise(500, 300, 5);
        let patch = blurred_noise(60, 30, 99);
        paste(&mut screen, &patch, 40, 40);
        paste(&mut screen, &patch, 380, 220);
        let own_window = Rect {
            x: 0.0,
            y: 0.0,
            width: 200.0,
            height: 150.0,
        };
        let found = find(&screen, &patch, &[own_window]).unwrap();
        assert_eq!((found.x, found.y), (380, 220));
    }

    #[test]
    fn absent_snippets_score_low() {
        let screen = blurred_noise(400, 300, 21);
        let template = blurred_noise(50, 30, 1234);
        let found = find(&screen, &template, &[]).unwrap();
        assert!(found.score < 0.8, "unexpected score {}", found.score);
    }

    #[test]
    fn flat_snippets_are_rejected() {
        let flat = Gray {
            width: 30,
            height: 20,
            pixels: vec![0.5; 600],
        };
        assert!(!has_detail(&flat));
        assert!(find(&blurred_noise(200, 100, 2), &flat, &[]).is_none());
    }

    #[test]
    fn png_density_gives_the_display_scale() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_pixel_dims(Some(png::PixelDimensions {
                xppu: 5669,
                yppu: 5669,
                unit: png::Unit::Meter,
            }));
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&[255, 255, 255, 0, 0, 0]).unwrap();
        }
        let decoded = decode_png(&bytes).unwrap();
        assert_eq!(decoded.scale, Some(2.0));
        assert_eq!((decoded.image.width, decoded.image.height), (2, 1));
        assert!((decoded.image.pixels[0] - 1.0).abs() < 1e-6);
        assert!(decoded.image.pixels[1].abs() < 1e-6);
    }

    #[test]
    fn resizing_preserves_matchability() {
        let screen = blurred_noise(400, 300, 17);
        let template = crop(&screen, 200, 100, 80, 40);
        let doubled = template.resized(160, 80);
        let restored = doubled.resized(80, 40);
        let found = find(&screen, &restored, &[]).unwrap();
        assert_eq!((found.x, found.y), (200, 100));
        assert!(found.score > 0.95);
    }
}
