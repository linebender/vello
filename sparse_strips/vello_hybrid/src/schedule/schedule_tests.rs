use super::test_support::{SceneCase, ScheduledCase};
use super::{IntermediateTextureAllocations, IntermediateTextureRequirements, ScheduleStorage};
use crate::filter::FILTER_ATLAS_PADDING;
use crate::target::{RootTarget, TextureParity};
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::geometry::SizeU16;
use vello_common::kurbo::Rect;
use vello_common::multi_atlas::AtlasError;
use vello_common::peniko::{BlendMode, Compose, Mix};

#[test]
fn intermediate_texture_requirements_validate_limit() {
    let allocations = |layer_pages, scratch| IntermediateTextureAllocations {
        layer_pages,
        scratch,
    };
    let base = IntermediateTextureRequirements {
        size: SizeU16::new(8),
        allocations: allocations([1, 1], true),
    };

    assert!(
        base.validate(IntermediateTextureAllocations::default(), None)
            .is_ok()
    );
    assert!(
        base.validate(IntermediateTextureAllocations::default(), Some(3))
            .is_ok()
    );
    assert!(base.validate(allocations([1, 0], true), Some(3)).is_ok());
    assert!(base.validate(allocations([0, 1], true), Some(3)).is_ok());
    assert!(base.validate(allocations([0, 2], false), Some(4)).is_ok());
    assert!(base.validate(allocations([0, 3], false), Some(4)).is_err());
    assert!(matches!(
        base.validate(allocations([2, 0], false), Some(3)),
        Err(AtlasError::NoSpaceAvailable)
    ));
    assert!(matches!(
        base.validate(allocations([1, 0], true), Some(2)),
        Err(AtlasError::NoSpaceAvailable)
    ));
    assert!(matches!(
        base.validate(allocations([0, 2], false), Some(3)),
        Err(AtlasError::NoSpaceAvailable)
    ));
    assert!(matches!(
        base.validate(allocations([2, 2], false), Some(4)),
        Err(AtlasError::NoSpaceAvailable)
    ));
}

fn blend_case() -> ScheduledCase {
    let mut case = SceneCase::new(32, 8);
    case.layer(|case| {
        case.draw(Rect::new(0.0, 0.0, 32.0, 8.0), 0.5);
        case.layer_with(
            None,
            Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
            None,
            |case| case.draw(Rect::new(8.0, 0.0, 16.0, 8.0), 0.5),
        );
    });
    case.schedule_root()
}

fn root_blend_case() -> SceneCase {
    let mut case = SceneCase::new(16, 8);
    case.layer_with(
        None,
        Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
        None,
        |case| case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5),
    );
    case
}

fn add_chain(case: &mut SceneCase, depth: usize) {
    case.layer(|case| {
        if depth == 1 {
            case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
        } else {
            add_chain(case, depth - 1);
        }
    });
}

fn add_blend_chain(case: &mut SceneCase, depth: usize) {
    case.layer_with(
        None,
        Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
        None,
        |case| {
            if depth == 1 {
                case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
            } else {
                add_blend_chain(case, depth - 1);
            }
        },
    );
}

fn add_tree(case: &mut SceneCase, depth: usize, children: usize) {
    case.layer(|case| {
        if depth == 1 {
            case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
        } else {
            for _ in 0..children {
                add_tree(case, depth - 1, children);
            }
        }
    });
}

fn chain_case(depth: usize) -> ScheduledCase {
    let mut case = SceneCase::new(8, 8);

    add_chain(&mut case, depth);
    case.schedule_root()
}

fn binding_case() -> ScheduledCase {
    let mut case = SceneCase::new(64, 64);
    case.layer(|case| case.draw(Rect::new(0.0, 0.0, 60.0, 60.0), 0.5));
    case.layer_with(
        None,
        Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
        Some(Filter::from_primitive(FilterPrimitive::Offset {
            dx: 0.0,
            dy: 0.0,
        })),
        |case| case.draw(Rect::new(0.0, 0.0, 60.0, 60.0), 0.5),
    );
    case.schedule(RootTarget::UserSurface, SizeU16::new(80), 8)
        .unwrap()
}

fn sibling_case(count: u16) -> SceneCase {
    let mut case = SceneCase::new(count * 4, 8);
    for index in 0..count {
        case.layer(|case| case.draw_at(f64::from(index * 4), 0.5));
    }
    case
}

fn offset_filter() -> Filter {
    Filter::from_primitive(FilterPrimitive::Offset { dx: 0.0, dy: 0.0 })
}

fn filter_page_size() -> SizeU16 {
    SizeU16::new(8 + 2 * FILTER_ATLAS_PADDING)
}

#[test]
fn empty_scene() {
    let scheduled = SceneCase::new(32, 32).schedule_root();

    assert!(scheduled.views().is_empty());
    assert_eq!(scheduled.page_counts(), [0, 0]);
    assert!(!scheduled.scratch_texture());
}

#[test]
fn root_draws() {
    let mut case = SceneCase::new(32, 8);
    for x in [0.0, 8.0, 16.0] {
        case.draw_at(x, 0.5);
    }

    let scheduled = case.schedule_root();
    let rounds_view = scheduled.views();

    assert_eq!(rounds_view.len(), 1);
    assert_eq!(rounds_view[0].root.x, [0, 8, 16]);
    assert!(!rounds_view[0].root.has_child_layer);
    assert_eq!(scheduled.page_counts(), [0, 0]);
}

#[test]
fn draw_order() {
    let mut case = SceneCase::new(32, 8);
    case.draw_at(0.0, 0.5);
    case.layer(|case| case.draw_at(8.0, 0.5));
    case.draw_at(16.0, 0.5);

    let scheduled = case.schedule_root();
    let rounds_view = scheduled.views();

    assert_eq!(rounds_view.len(), 1);
    assert_eq!(rounds_view[0].root.x, [0, 8, 16]);
    assert!(rounds_view[0].root.has_child_layer);
}

#[test]
fn opaque_root() {
    let mut case = SceneCase::new(32, 8);
    for x in [0.0, 8.0, 16.0] {
        case.draw_at(x, 1.0);
    }
    case.draw_at(24.0, 0.5);

    let user = case.schedule_root();
    assert_eq!(user.opaque_x(), [16, 8, 0]);
    assert_eq!(user.views()[0].root.x, [24]);

    let atlas = case
        .schedule(RootTarget::AtlasLayer, SizeU16::new(64), 8)
        .unwrap();
    assert!(atlas.opaque_x().is_empty());
    assert_eq!(atlas.views()[0].root.x, [0, 8, 16, 24]);
}

#[test]
fn simple_layer() {
    let mut case = SceneCase::new(32, 8);
    case.layer(|case| case.draw_at(8.0, 0.5));

    let scheduled = case.schedule_root();
    let rounds_view = scheduled.views();
    let round = &rounds_view[0];

    assert_eq!(scheduled.page_counts(), [0, 1]);
    assert_eq!(rounds_view.len(), 1);
    assert_eq!(round.odd.x.len(), 1);
    assert_eq!(round.root.x, [8]);
    assert!(round.root.has_child_layer);
    assert_eq!(round.clears[TextureParity::Odd.get_parity()].len(), 1);
}

#[test]
fn default_blend() {
    let mut case = SceneCase::new(16, 8);
    case.layer(|case| case.draw_at(4.0, 0.5));

    let scheduled = case.schedule_root();

    assert!(scheduled.views()[0].root.has_child_layer);
    assert!(scheduled.storage.buffers.blend_ops.is_empty());
}

#[test]
fn non_default_blend_into_layer() {
    let scheduled = blend_case();
    let rounds_view = scheduled.views();

    assert_eq!(rounds_view.len(), 1);
    assert_eq!(rounds_view[0].blend_passes, [0, 1]);
}

#[test]
fn clipped_away_blend() {
    let mut case = SceneCase::new(32, 8);
    case.layer(|case| {
        case.draw(Rect::new(0.0, 0.0, 32.0, 8.0), 0.5);
        case.layer_with(
            Some(Rect::new(24.0, 0.0, 32.0, 8.0)),
            Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
            Some(Filter::from_primitive(FilterPrimitive::Offset {
                dx: 0.0,
                dy: 0.0,
            })),
            |case| case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5),
        );
    });

    let scheduled = case.schedule_root();

    assert_eq!(scheduled.storage.buffers.filter_ops.len(), 1);
    assert!(scheduled.storage.buffers.blend_ops.is_empty());
    assert_eq!(scheduled.total_clears(), 3);
}

// Reproduces bug 2 from https://github.com/linebender/vello/pull/1759#issuecomment-5049523685.
#[test]
fn clipped_away_filtered_blend_under_atlas_pressure() {
    let mut case = SceneCase::new(16, 8);
    case.layer(|case| {
        case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
        // Filter layers are scheduled before their clip is applied. The disjoint clip therefore
        // makes the eventual blend a no-op without preventing the child filter from running.
        case.layer_with(
            Some(Rect::new(8.0, 0.0, 16.0, 8.0)),
            Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
            Some(offset_filter()),
            |case| case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5),
        );
    });

    // There is room for exactly one padded filter region per page. Allocating the parent therefore
    // advances the cursor before the clipped-away child is released.
    let scheduled = case
        .schedule(RootTarget::UserSurface, filter_page_size(), 2)
        .unwrap();

    assert_eq!(scheduled.storage.buffers.filter_ops.len(), 1);
    assert!(scheduled.storage.buffers.blend_ops.is_empty());
    let rounds_view = scheduled.views();
    assert_eq!(rounds_view.len(), 2);
    assert_eq!(rounds_view[1].odd.x.len(), 1);
    assert!(rounds_view[0].clears[TextureParity::Even.get_parity()].is_empty());
    assert_eq!(
        rounds_view[1].clears[TextureParity::Even.get_parity()].len(),
        1
    );
    assert_eq!(scheduled.total_clears(), 3);
}

#[test]
fn blend_release() {
    let scheduled = blend_case();
    let rounds_view = scheduled.views();
    let blend_round = rounds_view
        .iter()
        .find(|round| round.blend_passes[TextureParity::Odd.get_parity()] == 1)
        .unwrap();

    assert!(scheduled.scratch_texture());
    assert_eq!(
        blend_round.clears[TextureParity::Even.get_parity()].len(),
        1
    );
}

#[test]
fn root_blend_resources() {
    let scheduled = root_blend_case()
        .schedule(RootTarget::UserSurface, SizeU16::new(16), 3)
        .unwrap();

    // Root lands in the first odd layer, its child in the even one.
    assert_eq!(scheduled.page_counts(), [1, 1]);
    assert!(scheduled.scratch_texture());
}

#[test]
fn root_release() {
    let scheduled = root_blend_case()
        .schedule(RootTarget::UserSurface, SizeU16::new(16), 3)
        .unwrap();
    let rounds_view = scheduled.views();
    let root_round = rounds_view
        .iter()
        .find(|round| round.root.has_child_layer)
        .unwrap();

    assert_eq!(root_round.clears[TextureParity::Odd.get_parity()].len(), 1);
}

#[test]
fn root_blend_budget() {
    let case = root_blend_case();

    assert!(
        case.schedule(RootTarget::UserSurface, SizeU16::new(16), 3,)
            .is_ok()
    );
    assert!(matches!(
        case.schedule(RootTarget::UserSurface, SizeU16::new(16), 2,),
        Err(crate::RenderError::AtlasError(_))
    ));
}

#[test]
fn nested_parity() {
    let mut case = SceneCase::new(16, 8);
    case.layer(|case| {
        case.layer(|case| case.draw_at(4.0, 0.5));
    });

    let scheduled = case.schedule_root();
    let rounds_view = scheduled.views();
    let round = &rounds_view[0];

    assert_eq!(scheduled.page_counts(), [1, 1]);
    assert_eq!(rounds_view.len(), 1);
    assert_eq!(round.even.x.len(), 1);
    assert_eq!(round.odd.x.len(), 1);
    assert!(round.odd.has_child_layer);
    assert_eq!(round.root.x.len(), 1);
    assert!(round.root.has_child_layer);
    assert_eq!(round.clears[TextureParity::Even.get_parity()].len(), 1);
    assert_eq!(round.clears[TextureParity::Odd.get_parity()].len(), 1);
}

#[test]
fn even_child() {
    let scheduled = chain_case(2);
    let rounds_view = scheduled.views();

    assert_eq!(rounds_view.len(), 1);
    assert_eq!(rounds_view[0].even.x.len(), 1);
    assert!(rounds_view[0].odd.has_child_layer);
}

#[test]
fn odd_child() {
    let scheduled = chain_case(3);
    let rounds_view = scheduled.views();

    assert_eq!(rounds_view.len(), 2);
    assert_eq!(rounds_view[0].odd.x.len(), 1);
    assert!(rounds_view[0].even.x.is_empty());
    assert!(rounds_view[1].even.has_child_layer);
}

#[test]
fn draw_after_blend() {
    let mut case = SceneCase::new(32, 8);
    case.layer(|case| {
        case.draw_at(0.0, 0.5);
        case.layer_with(
            None,
            Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
            None,
            |case| case.draw_at(8.0, 0.5),
        );
        case.draw_at(24.0, 0.5);
    });

    let scheduled = case.schedule_root();
    let rounds_view = scheduled.views();

    assert_eq!(rounds_view[0].odd.x.len(), 1);
    assert_eq!(
        rounds_view[0].blend_passes[TextureParity::Odd.get_parity()],
        1
    );
    assert_eq!(rounds_view[1].odd.x.len(), 1);
}

#[test]
fn sibling_batch() {
    let mut case = SceneCase::new(32, 8);
    for x in [0.0, 8.0, 16.0] {
        case.layer(|case| case.draw_at(x, 0.5));
    }

    let scheduled = case
        .schedule(RootTarget::UserSurface, SizeU16::from_wh(16, 8), 1)
        .unwrap();
    let rounds_view = scheduled.views();
    let round = &rounds_view[0];

    assert_eq!(scheduled.page_counts(), [0, 1]);
    assert_eq!(rounds_view.len(), 1);
    assert_eq!(round.odd.x.len(), 3);
    assert_eq!(round.root.x, [0, 8, 16]);
    assert_eq!(round.clears[TextureParity::Odd.get_parity()].len(), 3);
}

#[test]
fn incompatible_siblings() {
    let scheduled = binding_case();
    let rounds_view = scheduled.views();

    // Root is blend target, so it lands in odd texture page 1.
    // Sibling layer lands in even texture page 1, and since it has
    // a filter we need a new page allocation in odd, so texthre page 2.
    assert_eq!(scheduled.page_counts(), [1, 2]);
    assert_eq!(rounds_view.len(), 3);
    // Round 0 binds the texture where the root is.
    assert_eq!(
        rounds_view[0].binding[TextureParity::Odd.get_parity()],
        Some(0)
    );
    // Round 1 binds the texture where the filter layer is.
    assert_eq!(
        rounds_view[1].binding[TextureParity::Odd.get_parity()],
        Some(1)
    );
    // Round 0 again binds to the root.
    assert_eq!(
        rounds_view[2].binding[TextureParity::Odd.get_parity()],
        Some(0)
    );
}

#[test]
fn filter_binding_conflict() {
    let scheduled = binding_case();
    let rounds_view = scheduled.views();
    let filter = scheduled.storage.buffers.filter_ops[0];

    assert_eq!(rounds_view[0].filter_passes, [0, 0]);
    assert_eq!(rounds_view[1].filter_passes, [1, 0]);
    for region in [filter.textures.original, filter.textures.temporary] {
        let target = region.target;
        assert_eq!(
            rounds_view[1].binding[target.texture_parity.get_parity()],
            Some(target.page_index)
        );
    }
}

#[test]
fn deep_reuse() {
    const DEPTH: usize = 12;
    let mut case = SceneCase::new(8, 8);
    add_chain(&mut case, DEPTH);

    let scheduled = case
        .schedule(RootTarget::UserSurface, SizeU16::new(8), 2)
        .unwrap();
    let rounds_view = scheduled.views();
    let layer_draws = rounds_view
        .iter()
        .map(|round| round.even.x.len() + round.odd.x.len())
        .sum::<usize>();

    assert!(rounds_view.len() > 1);
    assert_eq!(scheduled.page_counts(), [1, 1]);
    assert_eq!(layer_draws, DEPTH);
    assert_eq!(scheduled.total_clears(), DEPTH);
}

// For the next 3 cases: They show that our scheduler has the ability
// to render arbitrarily deeply nested layers as well as arbitrarily many
// sibling layers using just two textures, as long as they have
// at most one child.

#[test]
fn deeply_nested_layers() {
    for depth in 1..=32 {
        let mut case = SceneCase::new(8, 8);
        add_chain(&mut case, depth);

        let scheduled = case
            .schedule(RootTarget::UserSurface, SizeU16::new(8), 2)
            .unwrap();
        let textures = scheduled.page_counts().into_iter().sum::<usize>();

        assert!(textures <= 2, "depth {depth} used {textures} textures");
    }
}

#[test]
fn deeply_nested_blend_layers() {
    for depth in 1..=32 {
        let mut case = SceneCase::new(8, 8);
        add_blend_chain(&mut case, depth);

        let scheduled = case
            .schedule(RootTarget::UserSurface, SizeU16::new(8), 3)
            .unwrap();

        assert_eq!(
            (scheduled.scratch_texture(), scheduled.page_counts()),
            (true, [1, 1]),
            "unexpected resources at depth {depth}"
        );
    }
}

#[test]
fn wide_layers() {
    for count in 1..=32 {
        let scheduled = sibling_case(count)
            .schedule(RootTarget::UserSurface, SizeU16::from_wh(64, 8), 1)
            .unwrap();
        let rounds_view = scheduled.views();

        // Many sibling layers are batched into a single round, if atlas space
        // permits it.
        assert_eq!(rounds_view.len(), 1, "failed at width {count}");
        assert_eq!(
            rounds_view[0].odd.x.len(),
            usize::from(count),
            "missing child at width {count}"
        );
    }
}

#[test]
fn nested_children() {
    const CHILDREN: usize = 3;

    // If we have enough atlas space, even deeply and widely nested layer graphs can
    // be batched efficiently. The expected round count will be `layer_depth` / 2.
    // No additional pages need to be created as all layers fit.
    for (depth, expected_rounds) in (2..=6).zip([1, 2, 2, 3, 3]) {
        let mut case = SceneCase::new(8, 8);
        add_tree(&mut case, depth, CHILDREN);
        let scheduled = case
            .schedule(RootTarget::UserSurface, SizeU16::new(256), 2)
            .unwrap();
        let layers =
            (CHILDREN.pow(depth.try_into().expect("test depth fits in u32")) - 1) / (CHILDREN - 1);

        assert_eq!(scheduled.page_counts(), [1, 1], "failed at depth {depth}");
        assert_eq!(
            scheduled.views().len(),
            expected_rounds,
            "failed at depth {depth}"
        );
        assert_eq!(scheduled.total_clears(), layers);
    }
}

#[test]
fn nested_children_spilled() {
    const CHILDREN: usize = 3;

    // If our atlas dimensions are constrained, the scheduler will still find
    // a valid schedule and keep the number of allocated pages to a minimum.
    // However, this is at the cost of a larger round count.
    for (depth, expected_rounds) in (3..=6).zip([21, 39, 201, 363]) {
        let mut case = SceneCase::new(8, 8);
        add_tree(&mut case, depth, CHILDREN);
        let scheduled = case
            .schedule(RootTarget::UserSurface, SizeU16::new(8), 16)
            .unwrap();
        let layers =
            (CHILDREN.pow(depth.try_into().expect("test depth fits in u32")) - 1) / (CHILDREN - 1);

        assert_eq!(
            scheduled.page_counts(),
            [depth / 2, depth.div_ceil(2)],
            "failed at depth {depth}"
        );
        assert_eq!(
            scheduled.views().len(),
            expected_rounds,
            "failed at depth {depth}"
        );
        assert_eq!(scheduled.total_clears(), layers);
    }
}

#[test]
fn empty_layer() {
    let mut case = SceneCase::new(16, 8);
    case.layer(|_| {});

    let scheduled = case.schedule_root();

    assert!(scheduled.views().is_empty());
    assert_eq!(scheduled.page_counts(), [0, 0]);
}

#[test]
fn destructive_empty() {
    let mut case = SceneCase::new(32, 8);
    case.layer(|case| {
        case.draw(Rect::new(0.0, 0.0, 16.0, 8.0), 0.5);
        case.layer_with(
            None,
            Some(BlendMode::new(Mix::Normal, Compose::Clear)),
            None,
            |_| {},
        );
    });

    let scheduled = case.schedule_root();

    assert_eq!(scheduled.page_counts(), [1, 1]);
    assert!(scheduled.scratch_texture());
    assert_eq!(scheduled.storage.buffers.blend_ops.len(), 1);
}

#[test]
fn layer_clip() {
    let clip = Rect::new(8.0, 0.0, 16.0, 8.0);
    let mut clipped = SceneCase::new(32, 8);
    clipped.layer_with(Some(clip), None, None, |case| {
        case.draw(Rect::new(0.0, 0.0, 24.0, 8.0), 0.5);
    });
    let scheduled = clipped.schedule_root();
    let rounds_view = scheduled.views();
    assert_eq!(scheduled.page_counts(), [0, 1]);
    assert_eq!(rounds_view.len(), 1);
    assert_eq!(rounds_view[0].odd.x.len(), 1);
    assert!(rounds_view[0].root.has_child_layer);
    assert_eq!(
        rounds_view[0].clears[TextureParity::Odd.get_parity()].len(),
        1
    );

    let mut disjoint = SceneCase::new(32, 8);
    disjoint.layer_with(Some(Rect::new(24.0, 0.0, 32.0, 8.0)), None, None, |case| {
        case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
    });
    let scheduled = disjoint.schedule_root();
    assert!(scheduled.views().is_empty());
    assert_eq!(scheduled.page_counts(), [0, 0]);
}

#[test]
fn filter_layer() {
    let mut case = SceneCase::new(64, 16);
    let clip = Rect::new(16.0, 0.0, 24.0, 8.0);
    let filter = Filter::from_primitive(FilterPrimitive::Offset { dx: 8.0, dy: 0.0 });
    case.layer_with(Some(clip), None, Some(filter), |case| {
        case.draw(Rect::new(8.0, 0.0, 16.0, 8.0), 0.5);
    });

    let scheduled = case
        .schedule(RootTarget::UserSurface, SizeU16::new(128), 2)
        .unwrap();
    let rounds_view = scheduled.views();

    assert_eq!(scheduled.page_counts(), [1, 1]);
    assert_eq!(rounds_view.len(), 1);
    assert_eq!(rounds_view[0].odd.x.len(), 1);
    assert_eq!(rounds_view[0].filter_passes, [0, 1]);
    assert!(rounds_view[0].root.has_child_layer);
    assert_eq!(scheduled.total_clears(), 2);
}

#[test]
fn filter_round_resolving() {
    let mut case = SceneCase::new(8, 8);

    case.layer_with(None, None, Some(offset_filter()), |case| {
        case.layer(|case| case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5));
    });

    let scheduled = case
        .schedule(RootTarget::UserSurface, filter_page_size(), 2)
        .unwrap();
    let rounds_view = scheduled.views();
    let filter = scheduled.storage.buffers.filter_ops[0];

    // The filter source and its ping-pong temporary use opposite parities.
    assert_eq!(
        (
            filter.textures.original.target.texture_parity,
            filter.textures.temporary.target.texture_parity,
        ),
        (TextureParity::Odd, TextureParity::Even)
    );
    // No additional pages are created; the even page is reused after the child release.
    assert_eq!(scheduled.page_counts(), [1, 1]);
    assert_eq!(rounds_view.len(), 2);
    // The child is first drawn into an even texture in round 0.
    // The child is composed into the odd filter source **in round 0**. This is
    // possible because within a round, we first handle draws to even pages, then
    // odd pages.
    assert_eq!(rounds_view[0].odd.x.len(), 1);
    // The child allocation is released from the even page at the end of round 0.
    assert_eq!(
        rounds_view[0].clears[TextureParity::Even.get_parity()].len(),
        1
    );

    // The temporary is not available soon enough to filter in round 0.
    assert_eq!(rounds_view[0].filter_passes, [0, 0]);
    // After the even page becomes reusable, the filter runs in round 1.
    assert_eq!(rounds_view[1].filter_passes, [0, 1]);
}

#[test]
fn filter_siblings() {
    let mut case = SceneCase::new(8, 8);

    for _ in 0..2 {
        case.layer_with(None, None, Some(offset_filter()), |case| {
            case.draw(Rect::new(0.0, 0.0, 8.0, 8.0), 0.5);
        });
    }

    let scheduled = case
        .schedule(RootTarget::UserSurface, filter_page_size(), 2)
        .unwrap();

    // Sibling filter layers should also reuse pages.
    assert_eq!(scheduled.page_counts(), [1, 1]);
    assert_eq!(scheduled.storage.buffers.filter_ops.len(), 2);

    // Each filter clears its source allocation and its temporary allocation.
    assert_eq!(scheduled.total_clears(), 4);
}

#[test]
fn storage_reuse() {
    let mut first = SceneCase::new(64, 16);
    first.layer(|case| {
        case.draw(Rect::new(0.0, 0.0, 32.0, 8.0), 0.5);
        case.layer_with(
            Some(Rect::new(8.0, 0.0, 24.0, 8.0)),
            Some(BlendMode::new(Mix::Normal, Compose::Clear)),
            None,
            |case| case.draw(Rect::new(8.0, 0.0, 24.0, 8.0), 0.5),
        );
        case.layer_with(
            None,
            None,
            Some(Filter::from_primitive(FilterPrimitive::Offset {
                dx: 4.0,
                dy: 0.0,
            })),
            |case| case.draw(Rect::new(32.0, 0.0, 40.0, 8.0), 0.5),
        );
    });

    let mut storage = ScheduleStorage::default();
    let first_schedule = first
        .schedule_into(&mut storage, RootTarget::UserSurface, SizeU16::new(128), 4)
        .unwrap();

    assert!(!storage.buffers.draw_buffers.strips.is_empty());
    assert!(!storage.buffers.blend_ops.is_empty());
    assert!(!storage.buffers.blend_strips.is_empty());
    assert!(!storage.buffers.filter_ops.is_empty());
    assert!(!storage.filter_context.is_empty());
    assert!(!first_schedule.rounds.rounds.is_empty());

    let mut second = SceneCase::new(64, 16);
    second.draw_at(48.0, 0.5);
    let second_schedule = second
        .schedule_into(&mut storage, RootTarget::UserSurface, SizeU16::new(128), 4)
        .unwrap();

    assert_eq!(storage.buffers.draw_buffers.strips.len(), 1);
    assert!(storage.buffers.draw_buffers.opaque_strips.is_empty());
    assert!(storage.buffers.blend_ops.is_empty());
    assert!(storage.buffers.blend_strips.is_empty());
    assert!(storage.buffers.filter_ops.is_empty());
    assert!(storage.filter_context.is_empty());
    assert_eq!(second_schedule.rounds.rounds.len(), 1);
    let round = &second_schedule.rounds.rounds[0];
    let root = round
        .root_draw_pass(&storage.buffers, RootTarget::UserSurface)
        .unwrap();
    assert_eq!(root.strips.len(), 1);
    assert!(root.external_texture_runs.is_empty());
    assert_eq!(round.layer_passes(&storage.buffers).count(), 0);
}

#[test]
fn blend_is_constrained_to_parent_clip_bbox() {
    let mut case = SceneCase::new(32, 8);
    case.layer_with(Some(Rect::new(12.0, 0.0, 20.0, 8.0)), None, None, |case| {
        case.layer_with(
            None,
            Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
            None,
            |case| case.draw(Rect::new(4.0, 0.0, 16.0, 8.0), 0.5),
        );
    });
    let scheduled = case.schedule_root();
    let blend = scheduled.storage.buffers.blend_ops.first().unwrap();
    assert_eq!(blend.blend_bbox, blend.parent_region.layer_bbox);
    // This must not panic.
    let _ = crate::blend::GpuBlendInstance::new(blend, None, SizeU16::new(64));
}
