// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scene navigation controls for touch devices in the browser examples.

use std::{cell::Cell, rc::Rc};
use vello_common::kurbo::Point;
use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::{AddEventListenerOptions, Document, HtmlElement, TouchEvent};

/// Adds previous/next buttons, one-finger panning, and pinch zoom to `touch_target`.
///
/// `touch_target` gets `touch-action: none` so the browser does not claim gestures for scrolling,
/// zooming, or history navigation.
pub fn add_scene_navigation(
    touch_target: &HtmlElement,
    on_prev: impl Fn() + 'static,
    on_next: impl Fn() + 'static,
    on_pan: impl Fn(f64, f64) + 'static,
    on_zoom: impl Fn(f64, f64, f64) + 'static,
) {
    let on_prev: Rc<dyn Fn()> = Rc::new(on_prev);
    let on_next: Rc<dyn Fn()> = Rc::new(on_next);
    let on_pan: Rc<dyn Fn(f64, f64)> = Rc::new(on_pan);
    let on_zoom: Rc<dyn Fn(f64, f64, f64)> = Rc::new(on_zoom);
    let document = web_sys::window().unwrap().document().unwrap();

    let buttons = document
        .create_element("div")
        .unwrap()
        .dyn_into::<HtmlElement>()
        .unwrap();
    let style = buttons.style();
    style.set_property("position", "fixed").unwrap();
    // Sits below the demo selector that the deployed pages pin to the top-right corner.
    style.set_property("top", "52px").unwrap();
    style.set_property("right", "12px").unwrap();
    style.set_property("z-index", "1000").unwrap();
    style.set_property("display", "flex").unwrap();
    style.set_property("gap", "8px").unwrap();
    buttons
        .append_child(&create_button(
            &document,
            "←",
            "Previous scene",
            on_prev.clone(),
        ))
        .unwrap();
    buttons
        .append_child(&create_button(
            &document,
            "→",
            "Next scene",
            on_next.clone(),
        ))
        .unwrap();
    document.body().unwrap().append_child(&buttons).unwrap();

    add_touch_listeners(touch_target, on_pan, on_zoom);
}

fn create_button(
    document: &Document,
    text: &str,
    label: &str,
    on_click: Rc<dyn Fn()>,
) -> HtmlElement {
    let button = document
        .create_element("button")
        .unwrap()
        .dyn_into::<HtmlElement>()
        .unwrap();
    button.set_inner_text(text);
    button.set_title(label);
    button.set_attribute("aria-label", label).unwrap();
    let style = button.style();
    style.set_property("width", "44px").unwrap();
    style.set_property("height", "44px").unwrap();
    style.set_property("padding", "0").unwrap();
    style.set_property("border", "1px solid #555").unwrap();
    style.set_property("border-radius", "5px").unwrap();
    style.set_property("color", "white").unwrap();
    style
        .set_property("background", "rgba(0, 0, 0, 0.75)")
        .unwrap();
    style.set_property("font-size", "20px").unwrap();
    style.set_property("cursor", "pointer").unwrap();
    // Removes the double-tap-to-zoom delay when tapping the buttons repeatedly.
    style.set_property("touch-action", "manipulation").unwrap();

    let target = button.clone();
    let closure = Closure::wrap(Box::new(move || {
        // A focused button would also be activated by the Space key, which the
        // examples use to reset the view.
        target.blur().unwrap();
        on_click();
    }) as Box<dyn FnMut()>);
    button
        .add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())
        .unwrap();
    closure.forget();

    button
}

fn add_touch_listeners(
    target: &HtmlElement,
    on_pan: Rc<dyn Fn(f64, f64)>,
    on_zoom: Rc<dyn Fn(f64, f64, f64)>,
) {
    target.style().set_property("touch-action", "none").unwrap();

    let pan_position = Rc::new(Cell::new(None::<Point>));
    let pinch_distance = Rc::new(Cell::new(None::<f64>));
    let options = AddEventListenerOptions::new();
    options.set_passive(true);
    let add_listener = |event_type: &str, handler: Box<dyn FnMut(TouchEvent)>| {
        let closure = Closure::wrap(handler);
        target
            .add_event_listener_with_callback_and_add_event_listener_options(
                event_type,
                closure.as_ref().unchecked_ref(),
                &options,
            )
            .unwrap();
        closure.forget();
    };

    {
        let pan_position = pan_position.clone();
        let pinch_distance = pinch_distance.clone();
        add_listener(
            "touchstart",
            Box::new(move |event| {
                let touches = event.touches();
                match touches.length() {
                    1 => {
                        pan_position.set(touches.get(0).map(touch_point));
                        pinch_distance.set(None);
                    }
                    2 => {
                        pan_position.set(None);
                        pinch_distance.set(touch_pair(&touches).map(touch_distance));
                    }
                    _ => {
                        pan_position.set(None);
                        pinch_distance.set(None);
                    }
                }
            }),
        );
    }
    {
        let pan_position = pan_position.clone();
        let pinch_distance = pinch_distance.clone();
        add_listener(
            "touchend",
            Box::new(move |event| {
                let touches = event.touches();
                match touches.length() {
                    1 => {
                        pan_position.set(touches.get(0).map(touch_point));
                        pinch_distance.set(None);
                    }
                    2 => {
                        pan_position.set(None);
                        pinch_distance.set(touch_pair(&touches).map(touch_distance));
                    }
                    _ => {
                        pan_position.set(None);
                        pinch_distance.set(None);
                    }
                }
            }),
        );
    }
    {
        let pan_position = pan_position.clone();
        let pinch_distance = pinch_distance.clone();
        add_listener(
            "touchmove",
            Box::new(move |event| {
                let touches = event.touches();
                match touches.length() {
                    1 => {
                        pinch_distance.set(None);
                        let Some(current) = touches.get(0).map(touch_point) else {
                            return;
                        };
                        if let Some(previous) = pan_position.replace(Some(current)) {
                            on_pan(current.x - previous.x, current.y - previous.y);
                        }
                    }
                    2 => {
                        pan_position.set(None);
                        let Some((first, second)) = touch_pair(&touches) else {
                            return;
                        };
                        let distance = touch_distance((first.clone(), second.clone()));
                        if let Some(previous_distance) = pinch_distance.replace(Some(distance))
                            && previous_distance > 0.0
                        {
                            let center_x =
                                0.5 * (f64::from(first.client_x()) + f64::from(second.client_x()));
                            let center_y =
                                0.5 * (f64::from(first.client_y()) + f64::from(second.client_y()));
                            on_zoom(center_x, center_y, distance / previous_distance);
                        }
                    }
                    _ => {
                        pan_position.set(None);
                        pinch_distance.set(None);
                    }
                }
            }),
        );
    }
    add_listener(
        "touchcancel",
        Box::new(move |_| {
            pan_position.set(None);
            pinch_distance.set(None);
        }),
    );
}

fn touch_point(touch: web_sys::Touch) -> Point {
    Point::new(touch.client_x().into(), touch.client_y().into())
}

fn touch_pair(touches: &web_sys::TouchList) -> Option<(web_sys::Touch, web_sys::Touch)> {
    if touches.length() != 2 {
        return None;
    }
    Some((touches.get(0)?, touches.get(1)?))
}

fn touch_distance((first, second): (web_sys::Touch, web_sys::Touch)) -> f64 {
    let dx = f64::from(second.client_x() - first.client_x());
    let dy = f64::from(second.client_y() - first.client_y());
    dx.hypot(dy)
}
