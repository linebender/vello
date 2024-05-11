use std::{any::Any, marker::PhantomData};

use slotmap::{new_key_type, SecondaryMap, SlotMap};
use vello_encoding::Encoding;

use crate::{BufferProxy, FullShaders, ImageProxy, Recording, RenderParams, ResourceProxy};

use self::passes::RenderPass;

pub mod passes;

new_key_type! {
    pub struct PassId;
}

pub struct Pass<P: RenderPass>(PassId, PhantomData<P>);

pub struct RenderGraph {
    nodes: SlotMap<PassId, Box<dyn ErasedPass>>,
    dependencies: SecondaryMap<PassId, Vec<PassId>>,
    dependants: SecondaryMap<PassId, Vec<PassId>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        RenderGraph {
            nodes: SlotMap::with_key(),
            dependencies: SecondaryMap::new(),
            dependants: SecondaryMap::new(),
        }
    }

    pub fn insert_pass<
        D: IntoPassDependencies + 'static,
        P: RenderPass + 'static,
        F: Fn(D::Outputs) -> P + 'static,
    >(
        &mut self,
        dependencies: D,
        pass_builder: F,
    ) -> Pass<P> {
        let erased: PhantomPass<F, D, P> = PhantomPass {
            f: pass_builder,
            phantom: PhantomData,
        };
        let id = self.nodes.insert(Box::new(erased));
        self.dependants.insert(id, vec![]);
        let deps = dependencies.into_pass_dependencies();
        for dep in &deps {
            self.dependants[*dep].push(id);
        }
        self.dependencies.insert(id, deps);

        Pass(id, PhantomData)
    }

    pub fn process(
        &self,
        mut resources: ResourceManager,
        params: &RenderParams,
        shaders: &FullShaders,
        encoding: &Encoding,
        robust: bool,
    ) -> Option<Recording> {
        let mut recording = Recording::default();

        let mut stack = Vec::with_capacity(self.nodes.len());
        let mut counter = SecondaryMap::with_capacity(self.nodes.len());
        let mut result = Vec::with_capacity(self.nodes.len());

        for (id, d) in &self.dependencies {
            counter.insert(id, d.len());
            if d.len() == 0 {
                stack.push(id);
            }
        }

        while let Some(id) = stack.pop() {
            result.push(id);
            for &dependant in &self.dependants[id] {
                counter[dependant] -= 1;
                if counter[dependant] == 0 {
                    stack.push(dependant);
                }
            }
        }

        if result.len() != self.nodes.len() {
            return None;
        }

        let mut outputs: SecondaryMap<PassId, Box<dyn Any>> = SecondaryMap::new();

        for pass in result {
            let mut pass_recording = unsafe {
                self.nodes[pass].record(
                    pass,
                    &self.dependencies[pass],
                    &mut outputs,
                    PassContext {
                        resources: &mut resources,
                        params,
                        shaders,
                        encoding,
                        robust,
                    },
                )
            };
            recording.append(&mut pass_recording);
        }

        for (_, resource) in resources.resources.into_iter() {
            match resource {
                ManagableResource::Managed { proxy } => {
                    recording.free_resource(proxy);
                }
                ManagableResource::Imported => {}
            }
        }

        Some(recording)
    }
}

new_key_type! {
    pub struct ResourceId;
}

#[derive(Clone, Copy)]
pub struct Handle<T> {
    id: ResourceId,
    proxy: T,
}

impl Into<ResourceProxy> for Handle<ImageProxy> {
    fn into(self) -> ResourceProxy {
        ResourceProxy::Image(self.proxy)
    }
}
impl Into<ImageProxy> for Handle<ImageProxy> {
    fn into(self) -> ImageProxy {
        self.proxy
    }
}
impl Into<ResourceProxy> for Handle<BufferProxy> {
    fn into(self) -> ResourceProxy {
        ResourceProxy::Buffer(self.proxy)
    }
}
impl Into<BufferProxy> for Handle<BufferProxy> {
    fn into(self) -> BufferProxy {
        self.proxy
    }
}

enum ManagableResource {
    Managed { proxy: ResourceProxy },
    Imported,
}

pub struct ResourceManager {
    resources: SlotMap<ResourceId, ManagableResource>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            resources: SlotMap::with_key(),
        }
    }

    pub fn managed_image(&mut self, image: ImageProxy) -> Handle<ImageProxy> {
        let id = self.resources.insert(ManagableResource::Managed {
            proxy: image.into(),
        });
        Handle { id, proxy: image }
    }

    pub fn import_image(&mut self, image: ImageProxy) -> Handle<ImageProxy> {
        let id = self.resources.insert(ManagableResource::Imported);
        Handle { id, proxy: image }
    }

    pub fn managed_buffer(&mut self, buffer: BufferProxy) -> Handle<BufferProxy> {
        let id = self.resources.insert(ManagableResource::Managed {
            proxy: buffer.into(),
        });
        Handle { id, proxy: buffer }
    }

    pub fn import_buffer(&mut self, buffer: BufferProxy) -> Handle<BufferProxy> {
        let id = self.resources.insert(ManagableResource::Imported);
        Handle { id, proxy: buffer }
    }
}

pub trait IntoPassDependencies {
    type Outputs: Clone + Copy;

    fn into_pass_dependencies(self) -> Vec<PassId>;

    // SAFETY: assoc_data should match in length with outputs, returned any should have the type of the output
    unsafe fn outputs_map<'d, 'a, OD, OF>(assoc_data: &'d [OD], f: OF) -> Self::Outputs
    where
        OF: Fn(&'d OD) -> &'a dyn Any;
}

impl IntoPassDependencies for () {
    type Outputs = ();

    fn into_pass_dependencies(self) -> Vec<PassId> {
        vec![]
    }

    unsafe fn outputs_map<'d, 'a, OD, OF>(_assoc_data: &'d [OD], _f: OF) -> Self::Outputs
    where
        OF: Fn(&'d OD) -> &'a dyn Any,
    {
        ()
    }
}

macro_rules! impl_into_pass_dependencies {
    ( $(($generic:ident, $index:tt))+ ) => {
        impl<$($generic: RenderPass),+> IntoPassDependencies for ($(Pass<$generic>,)+) {
            type Outputs = ($($generic::Output,)+);

            #[inline]
            fn into_pass_dependencies(self) -> Vec<PassId> {
                vec![
                    $(
                        self.$index .0,
                    )+
                ]
            }

            unsafe fn outputs_map<'d, 'a, OD, OF>(assoc_data: &'d [OD], f: OF) -> Self::Outputs
            where
                OF: Fn(&'d OD) -> &'a dyn Any,
            {
                ($(
                    {
                        let any = f(&assoc_data[$index]);
                        unsafe {
                            any.downcast_ref::<$generic::Output>().unwrap_unchecked().clone()
                        }
                    },
                )+)
            }
        }
    };
}

impl_into_pass_dependencies!((A, 0));
impl_into_pass_dependencies!((A, 0)(B, 1));
impl_into_pass_dependencies!((A, 0)(B, 1)(C, 2));
impl_into_pass_dependencies!((A, 0)(B, 1)(C, 2)(D, 3));
impl_into_pass_dependencies!((A, 0)(B, 1)(C, 2)(D, 3)(E, 4));
impl_into_pass_dependencies!((A, 0)(B, 1)(C, 2)(D, 3)(E, 4)(F, 5));
impl_into_pass_dependencies!((A, 0)(B, 1)(C, 2)(D, 3)(E, 4)(F, 5)(G, 6));
impl_into_pass_dependencies!((A, 0)(B, 1)(C, 2)(D, 3)(E, 4)(F, 5)(G, 6)(H, 7));
impl_into_pass_dependencies!((A, 0)(B, 1)(C, 2)(D, 3)(E, 4)(F, 5)(G, 6)(H, 7)(I, 8));
impl_into_pass_dependencies!((A, 0)(B, 1)(C, 2)(D, 3)(E, 4)(F, 5)(G, 6)(H, 7)(I, 8)(J, 9));

struct PhantomPass<F, D, P> {
    f: F,
    phantom: PhantomData<(D, P)>,
}

trait ErasedPass {
    // SAFETY: make sure that all dependencies have already run and that the ids are valid and in the right order!
    unsafe fn record(
        &self,
        id: PassId,
        deps: &[PassId],
        outputs: &mut SecondaryMap<PassId, Box<dyn Any>>,
        cx: PassContext<'_>,
    ) -> Recording;
}

impl<F, D: IntoPassDependencies, P: RenderPass> ErasedPass for PhantomPass<F, D, P>
where
    F: Fn(D::Outputs) -> P,
{
    unsafe fn record(
        &self,
        id: PassId,
        deps: &[PassId],
        outputs: &mut SecondaryMap<PassId, Box<dyn Any>>,
        cx: PassContext<'_>,
    ) -> Recording {
        // SAFETY: user assures that everything is correct.
        let dep_outputs = D::outputs_map(deps, |dep| outputs[*dep].as_ref());
        let pass = (self.f)(dep_outputs);
        let (recording, output) = pass.record(cx);
        outputs.insert(id, Box::new(output));
        recording
    }
}

pub struct PassContext<'c> {
    pub resources: &'c mut ResourceManager,
    pub params: &'c RenderParams,
    pub shaders: &'c FullShaders,
    pub encoding: &'c Encoding,
    pub robust: bool,
}
