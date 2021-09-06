//! Vulkan implemenation of HAL trait.

use std::borrow::Cow;
use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;

use ash::extensions::{ext::DebugUtils, khr};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
use ash::{vk, Device, Entry, Instance};

use smallvec::SmallVec;

use crate::{BufferUsage, Error, GpuInfo, ImageLayout, SamplerParams, SubgroupSize, WorkgroupLimits};
use crate::backend::Device as DeviceTrait;


pub struct VkInstance {
    /// Retain the dynamic lib.
    #[allow(unused)]
    entry: Entry,
    instance: Instance,
    get_phys_dev_props: Option<vk::KhrGetPhysicalDeviceProperties2Fn>,
    vk_version: u32,
    _dbg_loader: Option<DebugUtils>,
    _dbg_callbk: Option<vk::DebugUtilsMessengerEXT>,
}

pub struct VkDevice {
    device: Arc<RawDevice>,
    physical_device: vk::PhysicalDevice,
    device_mem_props: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    qfi: u32,
    timestamp_period: f32,
    gpu_info: GpuInfo,
}

struct RawDevice {
    device: Device,
}

pub struct VkSurface {
    surface: vk::SurfaceKHR,
    surface_fn: khr::Surface,
}

pub struct VkSwapchain {
    swapchain: vk::SwapchainKHR,
    swapchain_fn: khr::Swapchain,

    present_queue: vk::Queue,

    acquisition_idx: usize,
    acquisition_semaphores: Vec<vk::Semaphore>, // same length as `images`
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
}

/// A handle to a buffer.
///
/// There is no lifetime tracking at this level; the caller is responsible
/// for destroying the buffer at the appropriate time.
pub struct Buffer {
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    // TODO: there should probably be a Buffer trait and this should be a method.
    pub size: u64,
}

pub struct Image {
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    extent: vk::Extent3D,
}

pub struct Pipeline {
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    max_textures: u32,
}

pub struct DescriptorSet {
    descriptor_set: vk::DescriptorSet,
}

pub struct CmdBuf {
    cmd_buf: vk::CommandBuffer,
    cmd_pool: vk::CommandPool,
    device: Arc<RawDevice>,
}

pub struct QueryPool {
    pool: vk::QueryPool,
    n_queries: u32,
}

#[derive(Clone, Copy)]
pub struct MemFlags(vk::MemoryPropertyFlags);

pub struct PipelineBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
    binding_flags: Vec<vk::DescriptorBindingFlags>,
    max_textures: u32,
}

pub struct DescriptorSetBuilder {
    buffers: Vec<vk::Buffer>,
    images: Vec<vk::ImageView>,
    textures: Vec<vk::ImageView>,
    sampler: vk::Sampler,
}

struct Extensions {
    exts: Vec<*const c_char>,
    exist_exts: Vec<vk::ExtensionProperties>,
}

struct Layers {
    layers: Vec<*const c_char>,
    exist_layers: Vec<vk::LayerProperties>,
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = &*p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity, message_type, message_id_name, message_id_number, message,
    );

    vk::FALSE
}

impl VkInstance {
    /// Create a new instance.
    ///
    /// There's more to be done to make this suitable for integration with other
    /// systems, but for now the goal is to make things simple.
    ///
    /// The caller is responsible for making sure that window which owns the raw window handle
    /// outlives the surface.
    pub fn new(
        window_handle: Option<&dyn raw_window_handle::HasRawWindowHandle>,
    ) -> Result<(VkInstance, Option<VkSurface>), Error> {
        unsafe {
            let app_name = CString::new("VkToy").unwrap();
            let entry = Entry::new()?;

            let mut layers = Layers::new(entry.enumerate_instance_layer_properties()?);
            if cfg!(debug_assertions) {
                layers
                    .try_add(CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());
            }

            let mut exts = Extensions::new(entry.enumerate_instance_extension_properties()?);
            let mut has_debug_ext = false;
            if cfg!(debug_assertions) {
                has_debug_ext = exts.try_add(DebugUtils::name());
            }
            // We'll need this to do runtime query of descriptor indexing.
            let has_phys_dev_props = exts.try_add(vk::KhrGetPhysicalDeviceProperties2Fn::name());
            if let Some(ref handle) = window_handle {
                for ext in ash_window::enumerate_required_extensions(*handle)? {
                    exts.try_add(ext);
                }
            }

            let supported_version = entry
                .try_enumerate_instance_version()?
                .unwrap_or(vk::make_version(1, 0, 0));
            let vk_version = if supported_version >= vk::make_version(1, 1, 0) {
                // We need Vulkan 1.1 to do subgroups; most other things can be extensions.
                vk::make_version(1, 1, 0)
            } else {
                vk::make_version(1, 0, 0)
            };

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(
                        &vk::ApplicationInfo::builder()
                            .application_name(&app_name)
                            .application_version(0)
                            .engine_name(&app_name)
                            .api_version(vk_version),
                    )
                    .enabled_layer_names(layers.as_ptrs())
                    .enabled_extension_names(exts.as_ptrs()),
                None,
            )?;

            let (_dbg_loader, _dbg_callbk) = if has_debug_ext {
                let dbg_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                    .pfn_user_callback(Some(vulkan_debug_callback));
                let dbg_loader = DebugUtils::new(&entry, &instance);
                let dbg_callbk = dbg_loader
                    .create_debug_utils_messenger(&dbg_info, None)
                    .unwrap();
                (Some(dbg_loader), Some(dbg_callbk))
            } else {
                (None, None)
            };

            let vk_surface = match window_handle {
                Some(handle) => Some(VkSurface {
                    surface: ash_window::create_surface(&entry, &instance, handle, None)?,
                    surface_fn: khr::Surface::new(&entry, &instance),
                }),
                None => None,
            };

            let get_phys_dev_props = if has_phys_dev_props {
                Some(vk::KhrGetPhysicalDeviceProperties2Fn::load(|name| {
                    std::mem::transmute(
                        entry.get_instance_proc_addr(instance.handle(), name.as_ptr()),
                    )
                }))
            } else {
                None
            };

            let vk_instance = VkInstance {
                entry,
                instance,
                get_phys_dev_props,
                vk_version,
                _dbg_loader,
                _dbg_callbk,
            };

            Ok((vk_instance, vk_surface))
        }
    }

    /// Create a device from the instance, suitable for compute, with an optional surface.
    ///
    /// # Safety
    ///
    /// The caller is responsible for making sure that the instance outlives the device
    /// and surface. We could enforce that, for example having an `Arc` of the raw instance,
    /// but for now keep things simple.
    pub unsafe fn device(&self, surface: Option<&VkSurface>) -> Result<VkDevice, Error> {
        let devices = self.instance.enumerate_physical_devices()?;
        let (pdevice, qfi) =
            choose_compute_device(&self.instance, &devices, surface).ok_or("no suitable device")?;

        let mut has_descriptor_indexing = false;
        if let Some(ref get_phys_dev_props) = self.get_phys_dev_props {
            let mut descriptor_indexing_features =
                vk::PhysicalDeviceDescriptorIndexingFeatures::builder();
            // See https://github.com/MaikKlein/ash/issues/325 for why we do this workaround.
            let mut features_v2 = vk::PhysicalDeviceFeatures2::default();
            features_v2.p_next =
                &mut descriptor_indexing_features as *mut _ as *mut std::ffi::c_void;
            get_phys_dev_props.get_physical_device_features2_khr(pdevice, &mut features_v2);
            has_descriptor_indexing = descriptor_indexing_features
                .shader_storage_image_array_non_uniform_indexing
                == vk::TRUE
                && descriptor_indexing_features.descriptor_binding_variable_descriptor_count
                    == vk::TRUE
                && descriptor_indexing_features.runtime_descriptor_array == vk::TRUE;
        }

        let queue_priorities = [1.0];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(qfi)
            .queue_priorities(&queue_priorities)
            .build()];

        let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
            .shader_storage_image_array_non_uniform_indexing(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true);

        let mut extensions = Extensions::new(
            self.instance
                .enumerate_device_extension_properties(pdevice)?,
        );
        if surface.is_some() {
            extensions.try_add(khr::Swapchain::name());
        }
        if has_descriptor_indexing {
            extensions.try_add(vk::KhrMaintenance3Fn::name());
            extensions.try_add(vk::ExtDescriptorIndexingFn::name());
        }
        let has_subgroup_size = self.vk_version >= vk::make_version(1, 1, 0)
            && extensions.try_add(vk::ExtSubgroupSizeControlFn::name());
        let has_memory_model = self.vk_version >= vk::make_version(1, 1, 0)
            && extensions.try_add(vk::KhrVulkanMemoryModelFn::name());
        let mut create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(extensions.as_ptrs());
        if has_descriptor_indexing {
            create_info = create_info.push_next(&mut descriptor_indexing);
        }
        let device = self.instance.create_device(pdevice, &create_info, None)?;

        let device_mem_props = self.instance.get_physical_device_memory_properties(pdevice);

        let queue_index = 0;
        let queue = device.get_device_queue(qfi, queue_index);

        let device = Arc::new(RawDevice { device });

        let props = self.instance.get_physical_device_properties(pdevice);
        let timestamp_period = props.limits.timestamp_period;
        let subgroup_size = if has_subgroup_size {
            let mut subgroup_props = vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT::default();
            let mut properties =
                vk::PhysicalDeviceProperties2::builder().push_next(&mut subgroup_props);
            self.instance
                .get_physical_device_properties2(pdevice, &mut properties);
            Some(SubgroupSize {
                min: subgroup_props.min_subgroup_size,
                max: subgroup_props.max_subgroup_size,
            })
        } else {
            None
        };

        // The question of when and when not to use staging buffers is complex, and this
        // is only a first approximation. Basically, it *must* be false when buffers can
        // be created with a memory type that is not host-visible. That is not guaranteed
        // here but is likely to be the case.
        //
        // I'm still investigating what should be done in systems with Resizable BAR.
        let use_staging_buffers = props.device_type != vk::PhysicalDeviceType::INTEGRATED_GPU;

        // TODO: finer grained query of specific subgroup info.
        let has_subgroups = self.vk_version >= vk::make_version(1, 1, 0);

        let workgroup_limits = WorkgroupLimits {
            max_invocations: props.limits.max_compute_work_group_invocations,
            max_size: props.limits.max_compute_work_group_size,
        };

        let gpu_info = GpuInfo {
            has_descriptor_indexing,
            has_subgroups,
            subgroup_size,
            workgroup_limits,
            has_memory_model,
            use_staging_buffers,
        };

        Ok(VkDevice {
            device,
            physical_device: pdevice,
            device_mem_props,
            qfi,
            queue,
            timestamp_period,
            gpu_info,
        })
    }

    pub unsafe fn swapchain(
        &self,
        width: usize,
        height: usize,
        device: &VkDevice,
        surface: &VkSurface,
    ) -> Result<VkSwapchain, Error> {
        let formats = surface
            .surface_fn
            .get_physical_device_surface_formats(device.physical_device, surface.surface)?;
        let surface_format = formats
            .iter()
            .map(|surface_fmt| match surface_fmt.format {
                vk::Format::UNDEFINED => {
                    vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8A8_UNORM, // most common format on desktop
                        color_space: surface_fmt.color_space,
                    }
                }
                _ => *surface_fmt,
            })
            .next()
            .ok_or("no surface format found")?;

        let capabilities = surface
            .surface_fn
            .get_physical_device_surface_capabilities(device.physical_device, surface.surface)?;

        let present_modes = surface
            .surface_fn
            .get_physical_device_surface_present_modes(device.physical_device, surface.surface)?;

        let present_mode = present_modes
            .into_iter()
            .find(|mode| mode == &vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let image_count = capabilities.min_image_count;
        let mut extent = capabilities.current_extent;
        if extent.width == u32::MAX || extent.height == u32::MAX {
            // We're deciding the size.
            extent.width = width as u32;
            extent.height = height as u32;
        }

        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain_fn = khr::Swapchain::new(&self.instance, &device.device.device);
        let swapchain = swapchain_fn.create_swapchain(&create_info, None)?;

        let images = swapchain_fn.get_swapchain_images(swapchain)?;
        let acquisition_semaphores = (0..images.len())
            .map(|_| device.create_semaphore())
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(VkSwapchain {
            swapchain,
            swapchain_fn,

            present_queue: device.queue,

            images,
            acquisition_semaphores,
            acquisition_idx: 0,
            extent,
        })
    }
}

impl crate::backend::Device for VkDevice {
    type Buffer = Buffer;
    type Image = Image;
    type CmdBuf = CmdBuf;
    type DescriptorSet = DescriptorSet;
    type Pipeline = Pipeline;
    type QueryPool = QueryPool;
    type Fence = vk::Fence;
    type Semaphore = vk::Semaphore;
    type PipelineBuilder = PipelineBuilder;
    type DescriptorSetBuilder = DescriptorSetBuilder;
    type Sampler = vk::Sampler;
    type ShaderSource = [u8];

    fn query_gpu_info(&self) -> GpuInfo {
        self.gpu_info.clone()
    }

    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Buffer, Error> {
        unsafe {
            let device = &self.device.device;
            let mut vk_usage = vk::BufferUsageFlags::empty();
            if usage.contains(BufferUsage::STORAGE) {
                vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
            }
            if usage.contains(BufferUsage::COPY_SRC) {
                vk_usage |= vk::BufferUsageFlags::TRANSFER_SRC;
            }
            if usage.contains(BufferUsage::COPY_DST) {
                vk_usage |= vk::BufferUsageFlags::TRANSFER_DST;
            }
            let buffer = device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let mem_requirements = device.get_buffer_memory_requirements(buffer);
            let mem_flags = memory_property_flags_for_usage(usage);
            let mem_type = find_memory_type(
                mem_requirements.memory_type_bits,
                mem_flags,
                &self.device_mem_props,
            )
            .unwrap(); // TODO: proper error
            let buffer_memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .allocation_size(mem_requirements.size)
                    .memory_type_index(mem_type),
                None,
            )?;
            device.bind_buffer_memory(buffer, buffer_memory, 0)?;
            Ok(Buffer {
                buffer,
                buffer_memory,
                size,
            })
        }
    }

    unsafe fn destroy_buffer(&self, buffer: &Self::Buffer) -> Result<(), Error> {
        let device = &self.device.device;
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.buffer_memory, None);
        Ok(())
    }

    unsafe fn create_image2d(&self, width: u32, height: u32) -> Result<Self::Image, Error> {
        let device = &self.device.device;
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        // TODO: maybe want to fine-tune these for different use cases, especially because we'll
        // want to add sampling for images and so on.
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let image = device.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let mem_requirements = device.get_image_memory_requirements(image);
        let mem_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let mem_type = find_memory_type(
            mem_requirements.memory_type_bits,
            mem_flags,
            &self.device_mem_props,
        )
        .unwrap(); // TODO: proper error
        let image_memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type),
            None,
        )?;
        device.bind_image_memory(image, image_memory, 0)?;
        let image_view = device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .image(image)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .build(),
            None,
        )?;
        Ok(Image {
            image,
            image_memory,
            image_view,
            extent,
        })
    }

    unsafe fn destroy_image(&self, image: &Self::Image) -> Result<(), Error> {
        let device = &self.device.device;
        device.destroy_image(image.image, None);
        device.destroy_image_view(image.image_view, None);
        device.free_memory(image.image_memory, None);
        Ok(())
    }

    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence, Error> {
        let device = &self.device.device;
        let mut flags = vk::FenceCreateFlags::empty();
        if signaled {
            flags |= vk::FenceCreateFlags::SIGNALED;
        }
        Ok(device.create_fence(&vk::FenceCreateInfo::builder().flags(flags).build(), None)?)
    }

    unsafe fn destroy_fence(&self, fence: Self::Fence) -> Result<(), Error> {
        let device = &self.device.device;
        device.destroy_fence(fence, None);
        Ok(())
    }

    unsafe fn create_semaphore(&self) -> Result<Self::Semaphore, Error> {
        let device = &self.device.device;
        Ok(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?)
    }

    unsafe fn wait_and_reset(&self, fences: Vec<&mut Self::Fence>) -> Result<(), Error> {
        let device = &self.device.device;
        let fences = fences.iter().map(|f| **f).collect::<SmallVec<[_; 4]>>();
        device.wait_for_fences(&fences, true, !0)?;
        device.reset_fences(&fences)?;
        Ok(())
    }

    unsafe fn get_fence_status(&self, fence: &mut Self::Fence) -> Result<bool, Error> {
        let device = &self.device.device;
        Ok(device.get_fence_status(*fence)?)
    }

    unsafe fn pipeline_builder(&self) -> PipelineBuilder {
        PipelineBuilder {
            bindings: Vec::new(),
            binding_flags: Vec::new(),
            max_textures: 0,
        }
    }

    unsafe fn descriptor_set_builder(&self) -> DescriptorSetBuilder {
        DescriptorSetBuilder {
            buffers: Vec::new(),
            images: Vec::new(),
            textures: Vec::new(),
            sampler: vk::Sampler::null(),
        }
    }

    fn create_cmd_buf(&self) -> Result<CmdBuf, Error> {
        unsafe {
            let device = &self.device.device;
            let cmd_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(self.qfi),
                None,
            )?;
            let cmd_buf = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            Ok(CmdBuf {
                cmd_buf,
                cmd_pool,
                device: self.device.clone(),
            })
        }
    }

    unsafe fn destroy_cmd_buf(&self, cmd_buf: CmdBuf) -> Result<(), Error> {
        let device = &self.device.device;
        device.destroy_command_pool(cmd_buf.cmd_pool, None);
        Ok(())
    }

    /// Create a query pool for timestamp queries.
    fn create_query_pool(&self, n_queries: u32) -> Result<QueryPool, Error> {
        unsafe {
            let device = &self.device.device;
            let pool = device.create_query_pool(
                &vk::QueryPoolCreateInfo::builder()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(n_queries),
                None,
            )?;
            Ok(QueryPool { pool, n_queries })
        }
    }

    unsafe fn fetch_query_pool(&self, pool: &Self::QueryPool) -> Result<Vec<f64>, Error> {
        let device = &self.device.device;
        let mut buf = vec![0u64; pool.n_queries as usize];
        device.get_query_pool_results(
            pool.pool,
            0,
            pool.n_queries,
            &mut buf,
            vk::QueryResultFlags::TYPE_64,
        )?;
        let ts0 = buf[0];
        let tsp = self.timestamp_period as f64 * 1e-9;
        let result = buf[1..]
            .iter()
            .map(|ts| ts.wrapping_sub(ts0) as f64 * tsp)
            .collect();
        Ok(result)
    }

    /// Run the command buffers.
    ///
    /// This submits the command buffer for execution. The provided fence
    /// is signalled when the execution is complete.
    unsafe fn run_cmd_bufs(
        &self,
        cmd_bufs: &[&CmdBuf],
        wait_semaphores: &[&Self::Semaphore],
        signal_semaphores: &[&Self::Semaphore],
        fence: Option<&mut Self::Fence>,
    ) -> Result<(), Error> {
        let device = &self.device.device;

        let fence = match fence {
            Some(fence) => *fence,
            None => vk::Fence::null(),
        };
        let wait_stages = wait_semaphores
            .iter()
            .map(|_| vk::PipelineStageFlags::ALL_COMMANDS)
            .collect::<SmallVec<[_; 4]>>();
        let cmd_bufs = cmd_bufs
            .iter()
            .map(|c| c.cmd_buf)
            .collect::<SmallVec<[_; 4]>>();
        let wait_semaphores = wait_semaphores
            .iter()
            .copied()
            .copied()
            .collect::<SmallVec<[_; 2]>>();
        let signal_semaphores = signal_semaphores
            .iter()
            .copied()
            .copied()
            .collect::<SmallVec<[_; 2]>>();
        device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::builder()
                .command_buffers(&cmd_bufs)
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&signal_semaphores)
                .build()],
            fence,
        )?;
        Ok(())
    }

    unsafe fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        dst: *mut u8,
        offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        let copy_size = size.try_into()?;
        let device = &self.device.device;
        let buf = device.map_memory(
            buffer.buffer_memory,
            offset,
            size,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(buf as *const u8, dst, copy_size);
        device.unmap_memory(buffer.buffer_memory);
        Ok(())
    }

    unsafe fn write_buffer(
        &self,
        buffer: &Buffer,
        contents: *const u8,
        offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        let copy_size = size.try_into()?;
        let device = &self.device.device;
        let buf = device.map_memory(
            buffer.buffer_memory,
            offset,
            size,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(contents, buf as *mut u8, copy_size);
        device.unmap_memory(buffer.buffer_memory);
        Ok(())
    }

    unsafe fn create_sampler(&self, params: SamplerParams) -> Result<Self::Sampler, Error> {
        let device = &self.device.device;
        let filter = match params {
            SamplerParams::Linear => vk::Filter::LINEAR,
            SamplerParams::Nearest => vk::Filter::NEAREST,
        };
        let sampler = device.create_sampler(
            &vk::SamplerCreateInfo::builder()
                .mag_filter(filter)
                .min_filter(filter)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .mip_lod_bias(0.0)
                .compare_op(vk::CompareOp::NEVER)
                .min_lod(0.0)
                .max_lod(0.0)
                .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
                .max_anisotropy(1.0)
                .anisotropy_enable(false),
            None,
        )?;
        Ok(sampler)
    }
}

impl crate::backend::CmdBuf<VkDevice> for CmdBuf {
    unsafe fn begin(&mut self) {
        self.device
            .device
            .begin_command_buffer(
                self.cmd_buf,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
    }

    unsafe fn finish(&mut self) {
        self.device.device.end_command_buffer(self.cmd_buf).unwrap();
    }

    unsafe fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        descriptor_set: &DescriptorSet,
        workgroup_count: (u32, u32, u32),
        _workgroup_size: (u32, u32, u32),
    ) {
        let device = &self.device.device;
        device.cmd_bind_pipeline(
            self.cmd_buf,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            self.cmd_buf,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline_layout,
            0,
            &[descriptor_set.descriptor_set],
            &[],
        );
        device.cmd_dispatch(
            self.cmd_buf,
            workgroup_count.0,
            workgroup_count.1,
            workgroup_count.2,
        );
    }

    /// Insert a pipeline barrier for all memory accesses.
    unsafe fn memory_barrier(&mut self) {
        let device = &self.device.device;
        device.cmd_pipeline_barrier(
            self.cmd_buf,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .build()],
            &[],
            &[],
        );
    }

    unsafe fn host_barrier(&mut self) {
        let device = &self.device.device;
        device.cmd_pipeline_barrier(
            self.cmd_buf,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .build()],
            &[],
            &[],
        );
    }

    unsafe fn image_barrier(
        &mut self,
        image: &Image,
        src_layout: ImageLayout,
        dst_layout: ImageLayout,
    ) {
        let device = &self.device.device;
        device.cmd_pipeline_barrier(
            self.cmd_buf,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::builder()
                .image(image.image)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .old_layout(map_image_layout(src_layout))
                .new_layout(map_image_layout(dst_layout))
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_MIP_LEVELS,
                })
                .build()],
        );
    }

    unsafe fn clear_buffer(&self, buffer: &Buffer, size: Option<u64>) {
        let device = &self.device.device;
        let size = size.unwrap_or(vk::WHOLE_SIZE);
        device.cmd_fill_buffer(self.cmd_buf, buffer.buffer, 0, size, 0);
    }

    unsafe fn copy_buffer(&self, src: &Buffer, dst: &Buffer) {
        let device = &self.device.device;
        let size = src.size.min(dst.size);
        device.cmd_copy_buffer(
            self.cmd_buf,
            src.buffer,
            dst.buffer,
            &[vk::BufferCopy::builder().size(size).build()],
        );
    }

    unsafe fn copy_image_to_buffer(&self, src: &Image, dst: &Buffer) {
        let device = &self.device.device;
        device.cmd_copy_image_to_buffer(
            self.cmd_buf,
            src.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst.buffer,
            &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,   // tight packing
                buffer_image_height: 0, // tight packing
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: src.extent,
            }],
        );
    }

    unsafe fn copy_buffer_to_image(&self, src: &Buffer, dst: &Image) {
        let device = &self.device.device;
        device.cmd_copy_buffer_to_image(
            self.cmd_buf,
            src.buffer,
            dst.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,   // tight packing
                buffer_image_height: 0, // tight packing
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: dst.extent,
            }],
        );
    }

    unsafe fn blit_image(&self, src: &Image, dst: &Image) {
        let device = &self.device.device;
        device.cmd_blit_image(
            self.cmd_buf,
            src.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src.extent.width as i32,
                        y: src.extent.height as i32,
                        z: src.extent.depth as i32,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst.extent.width as i32,
                        y: dst.extent.height as i32,
                        z: dst.extent.depth as i32,
                    },
                ],
            }],
            vk::Filter::LINEAR,
        );
    }

    unsafe fn reset_query_pool(&mut self, pool: &QueryPool) {
        let device = &self.device.device;
        device.cmd_reset_query_pool(self.cmd_buf, pool.pool, 0, pool.n_queries);
    }

    unsafe fn write_timestamp(&mut self, pool: &QueryPool, query: u32) {
        let device = &self.device.device;
        device.cmd_write_timestamp(
            self.cmd_buf,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            pool.pool,
            query,
        );
    }
}

impl crate::backend::PipelineBuilder<VkDevice> for PipelineBuilder {
    fn add_buffers(&mut self, n_buffers: u32) {
        let start = self.bindings.len() as u32;
        for i in 0..n_buffers {
            self.bindings.push(
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(start + i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            );
            self.binding_flags
                .push(vk::DescriptorBindingFlags::default());
        }
    }

    fn add_images(&mut self, n_images: u32) {
        let start = self.bindings.len() as u32;
        for i in 0..n_images {
            self.bindings.push(
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(start + i)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            );
            self.binding_flags
                .push(vk::DescriptorBindingFlags::default());
        }
    }

    fn add_textures(&mut self, n_images: u32) {
        let start = self.bindings.len() as u32;
        for i in 0..n_images {
            self.bindings.push(
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(start + i)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            );
            self.binding_flags
                .push(vk::DescriptorBindingFlags::default());
        }
        self.max_textures += n_images;
    }

    unsafe fn create_compute_pipeline(
        self,
        device: &VkDevice,
        code: &[u8],
    ) -> Result<Pipeline, Error> {
        let device = &device.device.device;
        let descriptor_set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&self.bindings)
                // It might be a slight optimization not to push this if max_textures = 0
                .push_next(
                    &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                        .binding_flags(&self.binding_flags)
                        .build(),
                ),
            None,
        )?;
        let descriptor_set_layouts = [descriptor_set_layout];

        // Create compute pipeline.
        let code_u32 = convert_u32_vec(code);
        let compute_shader_module = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&code_u32), None)?;
        let entry_name = CString::new("main").unwrap();
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts),
            None,
        )?;

        let pipeline = device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::builder()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::builder()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(compute_shader_module)
                            .name(&entry_name)
                            .build(),
                    )
                    .layout(pipeline_layout)
                    .build()],
                None,
            )
            .map_err(|(_pipeline, err)| err)?[0];
        Ok(Pipeline {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            max_textures: self.max_textures,
        })
    }
}

impl crate::backend::DescriptorSetBuilder<VkDevice> for DescriptorSetBuilder {
    fn add_buffers(&mut self, buffers: &[&Buffer]) {
        self.buffers.extend(buffers.iter().map(|b| b.buffer));
    }

    fn add_images(&mut self, images: &[&Image]) {
        self.images.extend(images.iter().map(|i| i.image_view));
    }

    fn add_textures(&mut self, images: &[&Image]) {
        self.textures.extend(images.iter().map(|i| i.image_view));
    }

    unsafe fn build(self, device: &VkDevice, pipeline: &Pipeline) -> Result<DescriptorSet, Error> {
        let device = &device.device.device;
        let mut descriptor_pool_sizes = Vec::new();
        if !self.buffers.is_empty() {
            descriptor_pool_sizes.push(
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(self.buffers.len() as u32)
                    .build(),
            );
        }
        if !self.images.is_empty() {
            descriptor_pool_sizes.push(
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(self.images.len() as u32)
                    .build(),
            );
        }
        if !self.textures.is_empty() {
            descriptor_pool_sizes.push(
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(self.textures.len() as u32)
                    .build(),
            );
        }
        let descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&descriptor_pool_sizes)
                .max_sets(1),
            None,
        )?;
        let descriptor_set_layouts = [pipeline.descriptor_set_layout];

        let descriptor_sets = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&descriptor_set_layouts),
            )
            .unwrap();
        let mut binding = 0;
        // Maybe one call to update_descriptor_sets with an array of descriptor_writes?
        for buf in &self.buffers {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[0])
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo::builder()
                        .buffer(*buf)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()])
                    .build()],
                &[],
            );
            binding += 1;
        }
        // maybe chain images and textures together; they're basically identical now
        for image in &self.images {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[0])
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo::builder()
                        .sampler(vk::Sampler::null())
                        .image_view(*image)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .build()])
                    .build()],
                &[],
            );
            binding += 1;
        }
        for image in &self.textures {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[0])
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo::builder()
                        .sampler(vk::Sampler::null())
                        .image_view(*image)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .build()])
                    .build()],
                &[],
            );
            binding += 1;
        }
        Ok(DescriptorSet {
            descriptor_set: descriptor_sets[0],
        })
    }
}

impl VkSwapchain {
    pub unsafe fn next(&mut self) -> Result<(usize, vk::Semaphore), Error> {
        let acquisition_semaphore = self.acquisition_semaphores[self.acquisition_idx];
        let (image_idx, _suboptimal) = self.swapchain_fn.acquire_next_image(
            self.swapchain,
            !0,
            acquisition_semaphore,
            vk::Fence::null(),
        )?;
        self.acquisition_idx = (self.acquisition_idx + 1) % self.acquisition_semaphores.len();

        Ok((image_idx as usize, acquisition_semaphore))
    }

    pub unsafe fn image(&self, idx: usize) -> Image {
        Image {
            image: self.images[idx],
            image_memory: vk::DeviceMemory::null(),
            image_view: vk::ImageView::null(),
            extent: vk::Extent3D {
                width: self.extent.width,
                height: self.extent.height,
                depth: 1,
            },
        }
    }

    pub unsafe fn present(
        &self,
        image_idx: usize,
        semaphores: &[&vk::Semaphore],
    ) -> Result<bool, Error> {
        let semaphores = semaphores
            .iter()
            .copied()
            .copied()
            .collect::<SmallVec<[_; 4]>>();
        Ok(self.swapchain_fn.queue_present(
            self.present_queue,
            &vk::PresentInfoKHR::builder()
                .swapchains(&[self.swapchain])
                .image_indices(&[image_idx as u32])
                .wait_semaphores(&semaphores)
                .build(),
        )?)
    }
}

impl Extensions {
    fn new(exist_exts: Vec<vk::ExtensionProperties>) -> Extensions {
        Extensions {
            exist_exts,
            exts: vec![],
        }
    }

    fn try_add(&mut self, ext: &'static CStr) -> bool {
        unsafe {
            if self
                .exist_exts
                .iter()
                .find(|x| CStr::from_ptr(x.extension_name.as_ptr()) == ext)
                .is_some()
            {
                self.exts.push(ext.as_ptr());
                true
            } else {
                false
            }
        }
    }

    fn as_ptrs(&self) -> &[*const c_char] {
        &self.exts
    }
}

impl Layers {
    fn new(exist_layers: Vec<vk::LayerProperties>) -> Layers {
        Layers {
            exist_layers,
            layers: vec![],
        }
    }

    fn try_add(&mut self, ext: &'static CStr) -> bool {
        unsafe {
            if self
                .exist_layers
                .iter()
                .find(|x| CStr::from_ptr(x.layer_name.as_ptr()) == ext)
                .is_some()
            {
                self.layers.push(ext.as_ptr());
                true
            } else {
                false
            }
        }
    }

    fn as_ptrs(&self) -> &[*const c_char] {
        &self.layers
    }
}

unsafe fn choose_compute_device(
    instance: &Instance,
    devices: &[vk::PhysicalDevice],
    surface: Option<&VkSurface>,
) -> Option<(vk::PhysicalDevice, u32)> {
    for pdevice in devices {
        let props = instance.get_physical_device_queue_family_properties(*pdevice);
        for (ix, info) in props.iter().enumerate() {
            // Check for surface presentation support
            if let Some(surface) = surface {
                if !surface
                    .surface_fn
                    .get_physical_device_surface_support(*pdevice, ix as u32, surface.surface)
                    .unwrap()
                {
                    continue;
                }
            }

            if info.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Some((*pdevice, ix as u32));
            }
        }
    }
    None
}

fn memory_property_flags_for_usage(usage: BufferUsage) -> vk::MemoryPropertyFlags {
    if usage.intersects(BufferUsage::MAP_READ | BufferUsage::MAP_WRITE) {
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    } else {
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    }
}

// This could get more sophisticated about asking for CACHED when appropriate, but is
// probably going to get replaced by a gpu-alloc solution anyway.
fn find_memory_type(
    memory_type_bits: u32,
    property_flags: vk::MemoryPropertyFlags,
    props: &vk::PhysicalDeviceMemoryProperties,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if (memory_type_bits & (1 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(property_flags)
        {
            return Some(i);
        }
    }
    None
}

fn convert_u32_vec(src: &[u8]) -> Vec<u32> {
    src.chunks(4)
        .map(|chunk| {
            let mut buf = [0; 4];
            buf.copy_from_slice(chunk);
            u32::from_le_bytes(buf)
        })
        .collect()
}

fn map_image_layout(layout: ImageLayout) -> vk::ImageLayout {
    match layout {
        ImageLayout::Undefined => vk::ImageLayout::UNDEFINED,
        ImageLayout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        ImageLayout::BlitSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ImageLayout::BlitDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ImageLayout::General => vk::ImageLayout::GENERAL,
        ImageLayout::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    }
}
