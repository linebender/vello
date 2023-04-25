/// Adapted from https://github.com/emilk/egui/blob/212656f3fc6b931b21eaad401e5cec2b0da93baa/crates/egui/src/input_state/touch_state.rs
use std::{collections::BTreeMap, fmt::Debug};

use vello::kurbo::{Point, Vec2};
use winit::event::{Touch, TouchPhase};

/// All you probably need to know about a multi-touch gesture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiTouchInfo {
    /// Number of touches (fingers) on the surface. Value is ≥ 2 since for a single touch no
    /// [`MultiTouchInfo`] is created.
    pub num_touches: usize,

    /// Proportional zoom factor (pinch gesture).
    /// * `zoom = 1`: no change
    /// * `zoom < 1`: pinch together
    /// * `zoom > 1`: pinch spread
    pub zoom_delta: f64,

    /// 2D non-proportional zoom factor (pinch gesture).
    ///
    /// For horizontal pinches, this will return `[z, 1]`,
    /// for vertical pinches this will return `[1, z]`,
    /// and otherwise this will return `[z, z]`,
    /// where `z` is the zoom factor:
    /// * `zoom = 1`: no change
    /// * `zoom < 1`: pinch together
    /// * `zoom > 1`: pinch spread
    pub zoom_delta_2d: Vec2,

    /// Rotation in radians. Moving fingers around each other will change this value. This is a
    /// relative value, comparing the orientation of fingers in the current frame with the previous
    /// frame. If all fingers are resting, this value is `0.0`.
    pub rotation_delta: f64,

    /// Relative movement (comparing previous frame and current frame) of the average position of
    /// all touch points. Without movement this value is `Vec2::ZERO`.
    ///
    /// Note that this may not necessarily be measured in screen points (although it _will_ be for
    /// most mobile devices). In general (depending on the touch device), touch coordinates cannot
    /// be directly mapped to the screen. A touch always is considered to start at the position of
    /// the pointer, but touch movement is always measured in the units delivered by the device,
    /// and may depend on hardware and system settings.
    pub translation_delta: Vec2,
    pub zoom_centre: Point,
}

/// The current state (for a specific touch device) of touch events and gestures.
#[derive(Clone)]
pub(crate) struct TouchState {
    /// Active touches, if any.
    ///
    /// TouchId is the unique identifier of the touch. It is valid as long as the finger/pen touches the surface. The
    /// next touch will receive a new unique ID.
    ///
    /// Refer to [`ActiveTouch`].
    active_touches: BTreeMap<u64, ActiveTouch>,

    /// If a gesture has been recognized (i.e. when exactly two fingers touch the surface), this
    /// holds state information
    gesture_state: Option<GestureState>,

    added_or_removed_touches: bool,
}

#[derive(Clone, Debug)]
struct GestureState {
    pinch_type: PinchType,
    previous: Option<DynGestureState>,
    current: DynGestureState,
}

/// Gesture data that can change over time
#[derive(Clone, Copy, Debug)]
struct DynGestureState {
    /// used for proportional zooming
    avg_distance: f64,
    /// used for non-proportional zooming
    avg_abs_distance2: Vec2,
    avg_pos: Point,
    heading: f64,
}

/// Describes an individual touch (finger or digitizer) on the touch surface. Instances exist as
/// long as the finger/pen touches the surface.
#[derive(Clone, Copy, Debug)]
struct ActiveTouch {
    /// Current position of this touch, in device coordinates (not necessarily screen position)
    pos: Point,
}

impl TouchState {
    pub fn new() -> Self {
        Self {
            active_touches: Default::default(),
            gesture_state: None,
            added_or_removed_touches: false,
        }
    }

    pub fn add_event(&mut self, event: &Touch) {
        let pos = Point::new(event.location.x, event.location.y);
        match event.phase {
            TouchPhase::Started => {
                self.active_touches.insert(event.id, ActiveTouch { pos });
                self.added_or_removed_touches = true;
            }
            TouchPhase::Moved => {
                if let Some(touch) = self.active_touches.get_mut(&event.id) {
                    touch.pos = Point::new(event.location.x, event.location.y);
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.active_touches.remove(&event.id);
                self.added_or_removed_touches = true;
            }
        }
    }

    pub fn end_frame(&mut self) {
        // This needs to be called each frame, even if there are no new touch events.
        // Otherwise, we would send the same old delta information multiple times:
        self.update_gesture();

        if self.added_or_removed_touches {
            // Adding or removing fingers makes the average values "jump". We better forget
            // about the previous values, and don't create delta information for this frame:
            if let Some(ref mut state) = &mut self.gesture_state {
                state.previous = None;
            }
        }
        self.added_or_removed_touches = false;
    }

    pub fn info(&self) -> Option<MultiTouchInfo> {
        self.gesture_state.as_ref().map(|state| {
            // state.previous can be `None` when the number of simultaneous touches has just
            // changed. In this case, we take `current` as `previous`, pretending that there
            // was no change for the current frame.
            let state_previous = state.previous.unwrap_or(state.current);

            let zoom_delta = if self.active_touches.len() > 1 {
                state.current.avg_distance / state_previous.avg_distance
            } else {
                1.
            };

            let zoom_delta2 = if self.active_touches.len() > 1 {
                match state.pinch_type {
                    PinchType::Horizontal => Vec2::new(
                        state.current.avg_abs_distance2.x / state_previous.avg_abs_distance2.x,
                        1.0,
                    ),
                    PinchType::Vertical => Vec2::new(
                        1.0,
                        state.current.avg_abs_distance2.y / state_previous.avg_abs_distance2.y,
                    ),
                    PinchType::Proportional => Vec2::new(zoom_delta, zoom_delta),
                }
            } else {
                Vec2::new(1.0, 1.0)
            };

            MultiTouchInfo {
                num_touches: self.active_touches.len(),
                zoom_delta,
                zoom_delta_2d: zoom_delta2,
                zoom_centre: state.current.avg_pos,
                rotation_delta: (state.current.heading - state_previous.heading),
                translation_delta: state.current.avg_pos - state_previous.avg_pos,
            }
        })
    }

    fn update_gesture(&mut self) {
        if let Some(dyn_state) = self.calc_dynamic_state() {
            if let Some(ref mut state) = &mut self.gesture_state {
                // updating an ongoing gesture
                state.previous = Some(state.current);
                state.current = dyn_state;
            } else {
                // starting a new gesture
                self.gesture_state = Some(GestureState {
                    pinch_type: PinchType::classify(&self.active_touches),
                    previous: None,
                    current: dyn_state,
                });
            }
        } else {
            // the end of a gesture (if there is any)
            self.gesture_state = None;
        }
    }

    /// `None` if less than two fingers
    fn calc_dynamic_state(&self) -> Option<DynGestureState> {
        let num_touches = self.active_touches.len();
        if num_touches == 0 {
            return None;
        }
        let mut state = DynGestureState {
            avg_distance: 0.0,
            avg_abs_distance2: Vec2::ZERO,
            avg_pos: Point::ZERO,
            heading: 0.0,
        };
        let num_touches_recip = 1. / num_touches as f64;

        // first pass: calculate force and center of touch positions:
        for touch in self.active_touches.values() {
            state.avg_pos.x += touch.pos.x;
            state.avg_pos.y += touch.pos.y;
        }
        state.avg_pos.x *= num_touches_recip;
        state.avg_pos.y *= num_touches_recip;

        // second pass: calculate distances from center:
        for touch in self.active_touches.values() {
            state.avg_distance += state.avg_pos.distance(touch.pos);
            state.avg_abs_distance2.x += (state.avg_pos.x - touch.pos.x).abs();
            state.avg_abs_distance2.y += (state.avg_pos.y - touch.pos.y).abs();
        }
        state.avg_distance *= num_touches_recip;
        state.avg_abs_distance2 *= num_touches_recip;

        // Calculate the direction from the first touch to the center position.
        // This is not the perfect way of calculating the direction if more than two fingers
        // are involved, but as long as all fingers rotate more or less at the same angular
        // velocity, the shortcomings of this method will not be noticed. One can see the
        // issues though, when touching with three or more fingers, and moving only one of them
        // (it takes two hands to do this in a controlled manner). A better technique would be
        // to store the current and previous directions (with reference to the center) for each
        // touch individually, and then calculate the average of all individual changes in
        // direction. But this approach cannot be implemented locally in this method, making
        // everything a bit more complicated.
        let first_touch = self.active_touches.values().next().unwrap();
        state.heading = (state.avg_pos - first_touch.pos).atan2();

        Some(state)
    }
}

impl Debug for TouchState {
    // This outputs less clutter than `#[derive(Debug)]`:
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (id, touch) in &self.active_touches {
            f.write_fmt(format_args!("#{:?}: {:#?}\n", id, touch))?;
        }
        f.write_fmt(format_args!("gesture: {:#?}\n", self.gesture_state))?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum PinchType {
    Horizontal,
    Vertical,
    Proportional,
}

impl PinchType {
    fn classify(touches: &BTreeMap<u64, ActiveTouch>) -> Self {
        // For non-proportional 2d zooming:
        // If the user is pinching with two fingers that have roughly the same Y coord,
        // then the Y zoom is unstable and should be 1.
        // Similarly, if the fingers are directly above/below each other,
        // we should only zoom on the Y axis.
        // If the fingers are roughly on a diagonal, we revert to the proportional zooming.

        if touches.len() == 2 {
            let mut touches = touches.values();
            let t0 = touches.next().unwrap().pos;
            let t1 = touches.next().unwrap().pos;

            let dx = (t0.x - t1.x).abs();
            let dy = (t0.y - t1.y).abs();

            if dx > 3.0 * dy {
                Self::Horizontal
            } else if dy > 3.0 * dx {
                Self::Vertical
            } else {
                Self::Proportional
            }
        } else {
            Self::Proportional
        }
    }
}
