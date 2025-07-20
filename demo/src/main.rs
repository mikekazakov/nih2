extern crate sdl3;

use nih::math::*;
use nih::render::*;
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
// use sdl3::pixels::Color;
// use sdl3::pixels::PixelFormat;
use sdl3::pixels::PixelFormatEnum;
use sdl3::rect::Rect;
use sdl3::surface::{Surface, SurfaceRef};

// use sdl3::{Error, pixels};
// use std::slice;

fn blit_to_window(buffer: &mut ColorBuffer, window: &sdl3::video::Window, event_pump: &sdl3::EventPump) {
    let width = buffer.width as u32;
    let height = buffer.height as u32;
    let pitch = (buffer.stride * 4) as u32;
    let buffer_surface =
        Surface::from_data(buffer.as_u8_slice_mut(), width, height, pitch, PixelFormatEnum::ABGR8888.into()).unwrap();

    let mut windows_surface = window.surface(&event_pump).unwrap();
    assert_eq!(windows_surface.width(), width);
    assert_eq!(windows_surface.height(), height);
    let rect = Rect::new(0, 0, width, height);
    buffer_surface.blit(rect, &mut windows_surface, rect).unwrap();
    windows_surface.finish().unwrap();
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;

    let mut window = video_subsystem
        .window("rust-sdl3 demo: Window", 800, 600)
        .resizable()
        .build()
        .map_err(|e| e.to_string())?;

    let mut buf = ColorBuffer::new(window.size().0 as usize, window.size().1 as usize);
    buf.fill(RGBA::new(255, 255, 0, 255));

    let mut tick = 0;

    let mut event_pump = sdl_context.event_pump().map_err(|e| e.to_string())?;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'running,
                _ => {}
            }
        }

        {
            let position = window.position();
            let size = window.size();
            let title = format!("Window - pos({}x{}), size({}x{}): {}", position.0, position.1, size.0, size.1, tick);
            window.set_title(&title).map_err(|e| e.to_string())?;

            tick += 1;

            buf.fill(RGBA::new((tick % 256) as u8, 255, 0, 255));
        }

        blit_to_window(&mut buf, &window, &event_pump);
    }

    Ok(())
}
