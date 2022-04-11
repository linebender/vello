use piet_scene::geometry::*;
use piet_scene::path::*;
use piet_scene::scene::*;
use piet_scene::{geometry::*, path::*, resource::ResourceContext, scene::*};

fn main() {
    let mut scene = Scene::default();
    let mut rcx = ResourceContext::new();
    let mut sb = build_scene(&mut scene, &mut rcx);

    sb.push_layer(Blend::default(), Rect::default().elements());

    let mut path = Path::new();
    let mut b = PathBuilder::new(&mut path);
    b.move_to(100., 100.);
    b.line_to(200., 200.);
    b.close_path();
    b.move_to(50., 50.);
    b.line_to(600., 150.);
    b.move_to(4., 2.);
    b.quad_to(8., 8., 9., 9.);
    b.close_path();
    println!("{:?}", path);
    for el in path.elements() {
        println!("{:?}", el);
    }
    //sb.push_layer(path.elements(), BlendMode::default());

    sb.push_layer(Blend::default(), [Element::MoveTo((0., 0.).into())]);
}
