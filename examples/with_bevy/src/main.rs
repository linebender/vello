use bevy::render::{Render, RenderSet};
use vello::fello::raw::tables::glyf::ToPathError;
use vello::kurbo::{Affine, BezPath, PathEl, Point, Rect, Shape};
use vello::peniko::{Color, Fill, Gradient, Join, Stroke};
use vello::{Renderer, RendererOptions, Scene, SceneBuilder, SceneFragment};

use bevy::{
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::RenderAssets,
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(VelloPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, bevy::window::close_on_esc)
        // .add_systems(Update, cube_rotator_system)
        .add_plugin(ExtractComponentPlugin::<VelloScene>::default())
        .add_systems(Update, render_fragment)
        .run()
}

#[derive(Resource)]
struct VelloRenderer(Renderer);

impl FromWorld for VelloRenderer {
    fn from_world(world: &mut World) -> Self {
        let device = world.get_resource::<RenderDevice>().unwrap();
        VelloRenderer(
            Renderer::new(
                device.wgpu_device(),
                &RendererOptions {
                    surface_format: None,
                },
            )
            .unwrap(),
        )
    }
}

struct VelloPlugin;

impl Plugin for VelloPlugin {
    fn build(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<VelloRenderer>();
        // This should probably use the render graph, but working out the dependencies there is awkward
        render_app.add_systems(Render, render_scenes.in_set(RenderSet::Render));
    }
}

fn render_scenes(
    mut renderer: ResMut<VelloRenderer>,
    mut scenes: Query<&VelloScene>,
    gpu_images: Res<RenderAssets<Image>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    for scene in &mut scenes {
        let Some(gpu_image) = gpu_images.get(&scene.1) else {
            continue;
        };

        let params = vello::RenderParams {
            base_color: vello::peniko::Color::TRANSPARENT,
            width: gpu_image.size.x as u32,
            height: gpu_image.size.y as u32,
        };
        renderer
            .0
            .render_to_texture(
                // WGPU
                device.wgpu_device(),
                // WGPU
                &*queue,
                // Vello
                &scene.0,
                // WGPU
                &gpu_image.texture_view,
                // Vello
                &params,
            )
            .unwrap();
    }
}

// Marks the main pass cube, to which the texture is applied.
#[derive(Component)]
struct MainPassCube;

#[derive(Component)]
pub struct VelloTarget(Handle<Image>);

#[derive(Component)]
// In the future, this will probably connect to the bevy heirarchy with an Affine component
pub struct VelloFragment(SceneFragment);

#[derive(Component)]
struct VelloScene(Scene, Handle<Image>);

impl ExtractComponent for VelloScene {
    type Query = (&'static VelloFragment, &'static VelloTarget);

    type Filter = ();

    type Out = Self;

    fn extract_component(
        (fragment, target): bevy::ecs::query::QueryItem<'_, Self::Query>,
    ) -> Option<Self> {
        let mut scene = Scene::default();
        let mut builder = SceneBuilder::for_scene(&mut scene);
        builder.append(&fragment.0, None);
        Some(Self(scene, target.0.clone()))
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let size = Extent3d {
        width: 512,
        height: 512,
        ..default()
    };

    // This is the texture that will be rendered to.
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        },
        ..default()
    };

    // fill image.data with zeroes
    image.resize(size);

    let image_handle = images.add(image);

    // Light
    // NOTE: Currently lights are shared between passes - see https://github.com/bevyengine/bevy/issues/3462
    commands.spawn(PointLightBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 0.0, 10.0)),
        ..default()
    });

    let cube_size = 4.0;
    let cube_handle = meshes.add(Mesh::from(shape::Box::new(cube_size, cube_size, cube_size)));

    // This material has the texture that has been rendered.
    let material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(image_handle.clone()),
        reflectance: 0.02,
        unlit: false,
        ..default()
    });

    // Main pass cube, with material containing the rendered first pass texture.
    commands.spawn((
        PbrBundle {
            mesh: cube_handle,
            material: material_handle,
            transform: Transform::from_xyz(0.0, 0.0, 1.5)
                .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 5.0)),
            ..default()
        },
        MainPassCube,
    ));

    // The main pass camera.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    commands.spawn((
        VelloFragment(SceneFragment::default()),
        VelloTarget(image_handle),
    ));
}

/// Rotates the outer cube (main pass)
fn cube_rotator_system(time: Res<Time>, mut query: Query<&mut Transform, With<MainPassCube>>) {
    for mut transform in &mut query {
        transform.rotate_x(1.0 * time.delta_seconds());
        transform.rotate_y(0.7 * time.delta_seconds());
    }
}

fn render_fragment(mut fragment: Query<&mut VelloFragment>, mut time: Res<Time>) {
    let mut fragment = fragment.single_mut();
    let mut builder = SceneBuilder::for_fragment(&mut fragment.0);
    render_brush_transform(&mut builder, time.elapsed_seconds_f64());
    // *time += 1;
}

fn render_brush_transform(sb: &mut SceneBuilder, time: f64) {
    // let th = (std::f64::consts::PI / 180.0) * (i as f64);
    // let linear = Gradient::new_linear((0.0, 0.0), (0.0, 200.0)).with_stops([
    //     Color::RED,
    //     Color::GREEN,
    //     Color::BLUE,
    // ]);

    let fill_color = Color::CHARTREUSE;
    let stroke_color = Color::ORCHID;

    let start_point: Point = Point::default();

    let origin_curve: PathEl = PathEl::CurveTo(
        Point::new(0.0, 100.0),
        Point::new(100.0, 0.0),
        Point::new(100.0, 100.0),
    );

    let percent: f64 = time.sin().mul_add(0.5, 0.5);
    let new_curve: PathEl;

    match origin_curve {
        PathEl::CurveTo(b, c, d) => {
            let ab = Point::lerp(start_point, b, percent);
            let bc = Point::lerp(b, c, percent);
            let cd = Point::lerp(c, d, percent);

            let abc = Point::lerp(ab, bc, percent);
            let bcd = Point::lerp(bc, cd, percent);

            let abcd = Point::lerp(abc, bcd, percent);

            new_curve = PathEl::CurveTo(ab, abc, abcd);
        }
        _ => {
            return;
        }
    }

    let path = BezPath::from_vec(vec![PathEl::MoveTo(start_point), new_curve]);

    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &fill_color,
        // &linear,
        None,
        // Some(around_center(Affine::rotate(th), Point::new(150.0, 150.0))),
        &Rect::from_origin_size(Point::default(), (300.0, 300.0)),
    );
    sb.stroke(
        &Stroke::new(10.0),
        Affine::translate((106.0, 106.0)),
        &stroke_color,
        // &linear,
        None,
        // Some(around_center(
        //     Affine::rotate(th + std::f64::consts::PI / 2.),
        //     Point::new(176.5, 176.5),
        // )),
        // &Rect::from_origin_size(Point::new(53.0, 53.0), (406.0, 406.0)),
        &path.elements(),
    );
}

fn around_center(xform: Affine, center: Point) -> Affine {
    Affine::translate(center.to_vec2()) * xform * Affine::translate(-center.to_vec2())
}

// TODO
pub fn interpolate_bezpath(path: BezPath, t: f64) {}

pub fn interpolate_pathel(path0: PathEl, path1: PathEl, t: f64) -> Option<PathEl> {
    let a: Point;
    match path0 {
        PathEl::MoveTo(p) => a = p,
        PathEl::LineTo(p) => a = p,
        PathEl::QuadTo(_, p) => a = p,
        PathEl::CurveTo(_, _, p) => a = p,
        PathEl::ClosePath => return None,
    }

    let path: PathEl;

    match path1 {
        PathEl::MoveTo(_) => return None,
        PathEl::LineTo(b) => path = PathEl::LineTo(Point::lerp(a, b, t)),
        PathEl::QuadTo(b, c) => {
            let ab: Point = Point::lerp(a, b, t);
            let bc: Point = Point::lerp(b, c, t);

            let abc: Point = Point::lerp(ab, bc, t);

            path = PathEl::QuadTo(ab, abc);
        }
        PathEl::CurveTo(b, c, d) => {
            let ab = Point::lerp(a, b, t);
            let bc = Point::lerp(b, c, t);
            let cd = Point::lerp(c, d, t);

            let abc = Point::lerp(ab, bc, t);
            let bcd = Point::lerp(bc, cd, t);

            let abcd = Point::lerp(abc, bcd, t);

            path = PathEl::CurveTo(ab, abc, abcd);
        }
        PathEl::ClosePath => return None,
    }

    Some(path)
}
