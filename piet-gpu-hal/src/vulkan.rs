//! Vulkan implemenation of HAL trait.

use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::sync::Arc;

use ash::extensions::{ext::DebugUtils, khr};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, Device, Entry, Instance};
use once_cell::sync::Lazy;

use crate::{Device as DeviceTrait, Error, ImageLayout};

pub struct VkInstance {
    /// Retain the dynamic lib.
    #[allow(unused)]
    entry: Entry,
    instance: Instance,
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
    size: u64,
}

pub struct Image {
    image: vk::Image,
    // Not used now but probably will be for destruction.
    _image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    extent: vk::Extent3D,
}

pub struct Pipeline {
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
}

pub struct DescriptorSet {
    descriptor_set: vk::DescriptorSet,
}

pub struct CmdBuf {
    cmd_buf: vk::CommandBuffer,
    device: Arc<RawDevice>,
}

pub struct QueryPool {
    pool: vk::QueryPool,
    n_queries: u32,
}

#[derive(Clone, Copy)]
pub struct MemFlags(vk::MemoryPropertyFlags);

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

static LAYERS: Lazy<Vec<&'static CStr>> = Lazy::new(|| {
    let mut layers: Vec<&'static CStr> = vec![];
    if cfg!(debug_assertions) {
        layers.push(CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());
    }
    layers
});

static EXTS: Lazy<Vec<&'static CStr>> = Lazy::new(|| {
    let mut exts: Vec<&'static CStr> = vec![];
    if cfg!(debug_assertions) {
        exts.push(DebugUtils::name());
    }
    exts
});

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

            let exist_layers = entry.enumerate_instance_layer_properties()?;
            let layers = LAYERS
                .iter()
                .filter_map(|&lyr| {
                    exist_layers
                        .iter()
                        .find(|x| CStr::from_ptr(x.layer_name.as_ptr()) == lyr)
                        .map(|_| lyr.as_ptr())
                        .or_else(|| {
                            println!(
                                "Unable to find layer: {}, have you installed the Vulkan SDK?",
                                lyr.to_string_lossy()
                            );
                            None
                        })
                })
                .collect::<Vec<_>>();

            let exist_exts = entry.enumerate_instance_extension_properties()?;
            let mut exts = EXTS
                .iter()
                .filter_map(|&ext| {
                    exist_exts
                        .iter()
                        .find(|x| CStr::from_ptr(x.extension_name.as_ptr()) == ext)
                        .map(|_| ext.as_ptr())
                        .or_else(|| {
                            println!(
                                "Unable to find extension: {}, have you installed the Vulkan SDK?",
                                ext.to_string_lossy()
                            );
                            None
                        })
                })
                .collect::<Vec<_>>();

            let surface_extensions = match window_handle {
                Some(ref handle) => ash_window::enumerate_required_extensions(*handle)?,
                None => vec![],
            };
            for extension in surface_extensions {
                exts.push(extension.as_ptr());
            }

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(
                        &vk::ApplicationInfo::builder()
                            .application_name(&app_name)
                            .application_version(0)
                            .engine_name(&app_name)
                            .api_version(vk::make_version(1, 0, 0)),
                    )
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&exts),
                None,
            )?;

            let (_dbg_loader, _dbg_callbk) = if cfg!(debug_assertions) {
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

            let vk_instance = VkInstance {
                entry,
                instance,
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

        let queue_priorities = [1.0];
        let queue_create_infos = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(qfi)
            .queue_priorities(&queue_priorities)
            .build()];
        let extensions = match surface {
            Some(_) => vec![khr::Swapchain::name().as_ptr()],
            None => vec![],
        };
        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions)
            .build();
        let device = self.instance.create_device(pdevice, &create_info, None)?;

        let device_mem_props = self.instance.get_physical_device_memory_properties(pdevice);

        let queue_index = 0;
        let queue = device.get_device_queue(qfi, queue_index);

        let device = Arc::new(RawDevice { device });

        let props = self.instance.get_physical_device_properties(pdevice);
        let timestamp_period = props.limits.timestamp_period;

        Ok(VkDevice {
            device,
            physical_device: pdevice,
            device_mem_props,
            qfi,
            queue,
            timestamp_period,
        })
    }

    pub unsafe fn swapchain(
        &self,
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

        let image_count = 2; // TODO
        let extent = capabilities.current_extent; // TODO: wayland for example will complain here ..

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

impl crate::Device for VkDevice {
    type Buffer = Buffer;
    type Image = Image;
    type CmdBuf = CmdBuf;
    type DescriptorSet = DescriptorSet;
    type Pipeline = Pipeline;
    type QueryPool = QueryPool;
    type MemFlags = MemFlags;
    type Fence = vk::Fence;
    type Semaphore = vk::Semaphore;

    fn create_buffer(&self, size: u64, mem_flags: MemFlags) -> Result<Buffer, Error> {
        unsafe {
            let device = &self.device.device;
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
            let mem_type = find_memory_type(
                mem_requirements.memory_type_bits,
                mem_flags.0,
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

    unsafe fn create_image2d(
        &self,
        width: u32,
        height: u32,
        mem_flags: Self::MemFlags,
    ) -> Result<Self::Image, Error> {
        let device = &self.device.device;
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
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
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC) // write in compute and blit src
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let mem_requirements = device.get_image_memory_requirements(image);
        let mem_type = find_memory_type(
            mem_requirements.memory_type_bits,
            mem_flags.0,
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
            _image_memory: image_memory,
            image_view,
            extent,
        })
    }

    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence, Error> {
        let device = &self.device.device;
        let mut flags = vk::FenceCreateFlags::empty();
        if signaled {
            flags |= vk::FenceCreateFlags::SIGNALED;
        }
        Ok(device.create_fence(&vk::FenceCreateInfo::builder().flags(flags).build(), None)?)
    }

    unsafe fn create_semaphore(&self) -> Result<Self::Semaphore, Error> {
        let device = &self.device.device;
        Ok(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?)
    }

    unsafe fn wait_and_reset(&self, fences: &[Self::Fence]) -> Result<(), Error> {
        let device = &self.device.device;
        device.wait_for_fences(fences, true, !0)?;
        device.reset_fences(fences)?;
        Ok(())
    }

    /// This creates a pipeline that runs over the buffer.
    ///
    /// The descriptor set layout is just some number of storage buffers and storage images (this might change).
    unsafe fn create_simple_compute_pipeline(
        &self,
        code: &[u8],
        n_buffers: u32,
        n_images: u32,
    ) -> Result<Pipeline, Error> {
        let device = &self.device.device;
        let mut bindings = Vec::new();
        for i in 0..n_buffers {
            bindings.push(
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            );
        }
        for i in n_buffers..n_buffers + n_images {
            bindings.push(
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            );
        }
        let descriptor_set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings),
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
        })
    }

    unsafe fn create_descriptor_set(
        &self,
        pipeline: &Pipeline,
        bufs: &[&Buffer],
        images: &[&Image],
    ) -> Result<DescriptorSet, Error> {
        let device = &self.device.device;
        let mut descriptor_pool_sizes = Vec::new();
        if !bufs.is_empty() {
            descriptor_pool_sizes.push(
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(bufs.len() as u32)
                    .build(),
            );
        }
        if !images.is_empty() {
            descriptor_pool_sizes.push(
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(images.len() as u32)
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
        for (i, buf) in bufs.iter().enumerate() {
            let buf_info = vk::DescriptorBufferInfo::builder()
                .buffer(buf.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[0])
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[buf_info])
                    .build()],
                &[],
            );
        }
        for (i, image) in images.iter().enumerate() {
            let binding = i + bufs.len();
            let image_info = vk::DescriptorImageInfo::builder()
                .sampler(vk::Sampler::null())
                .image_view(image.image_view)
                .image_layout(vk::ImageLayout::GENERAL)
                .build();
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[0])
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[image_info])
                    .build()],
                &[],
            );
        }
        Ok(DescriptorSet {
            descriptor_set: descriptor_sets[0],
        })
    }

    fn create_cmd_buf(&self) -> Result<CmdBuf, Error> {
        unsafe {
            let device = &self.device.device;
            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(self.qfi),
                None,
            )?;
            let cmd_buf = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            Ok(CmdBuf {
                cmd_buf,
                device: self.device.clone(),
            })
        }
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

    unsafe fn reap_query_pool(&self, pool: &Self::QueryPool) -> Result<Vec<f64>, Error> {
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

    /// Run the command buffer.
    ///
    /// This version simply blocks until it's complete.
    unsafe fn run_cmd_buf(
        &self,
        cmd_buf: &CmdBuf,
        wait_semaphores: &[Self::Semaphore],
        signal_semaphores: &[Self::Semaphore],
        fence: Option<&Self::Fence>,
    ) -> Result<(), Error> {
        let device = &self.device.device;

        let fence = match fence {
            Some(fence) => *fence,
            None => vk::Fence::null(),
        };
        let wait_stages = wait_semaphores
            .iter()
            .map(|_| vk::PipelineStageFlags::ALL_COMMANDS)
            .collect::<Vec<_>>();
        device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::builder()
                .command_buffers(&[cmd_buf.cmd_buf])
                .wait_semaphores(wait_semaphores)
                .signal_semaphores(signal_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .build()],
            fence,
        )?;
        Ok(())
    }

    unsafe fn read_buffer<T: Sized>(
        &self,
        buffer: &Buffer,
        result: &mut Vec<T>,
    ) -> Result<(), Error> {
        let device = &self.device.device;
        let size = buffer.size as usize / std::mem::size_of::<T>();
        let buf = device.map_memory(
            buffer.buffer_memory,
            0,
            size as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        if size > result.len() {
            result.reserve(size - result.len());
        }
        std::ptr::copy_nonoverlapping(buf as *const T, result.as_mut_ptr(), size);
        result.set_len(size);
        device.unmap_memory(buffer.buffer_memory);
        Ok(())
    }

    unsafe fn write_buffer<T: Sized>(&self, buffer: &Buffer, contents: &[T]) -> Result<(), Error> {
        let device = &self.device.device;
        let buf = device.map_memory(
            buffer.buffer_memory,
            0,
            std::mem::size_of_val(contents) as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(contents.as_ptr(), buf as *mut T, contents.len());
        device.unmap_memory(buffer.buffer_memory);
        Ok(())
    }
}

impl crate::CmdBuf<VkDevice> for CmdBuf {
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
        size: (u32, u32, u32),
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
        device.cmd_dispatch(self.cmd_buf, size.0, size.1, size.2);
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

    unsafe fn clear_buffer(&self, buffer: &Buffer) {
        let device = &self.device.device;
        device.cmd_fill_buffer(self.cmd_buf, buffer.buffer, 0, vk::WHOLE_SIZE, 0);
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

impl crate::MemFlags for MemFlags {
    fn device_local() -> Self {
        MemFlags(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    fn host_coherent() -> Self {
        MemFlags(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
    }
}

impl VkSwapchain {
    pub unsafe fn next(&mut self) -> Result<(usize, vk::Semaphore), Error> {
        let acquisition_semaphore = self.acquisition_semaphores[self.acquisition_idx];
        let (image_idx, _suboptimal) = self.swapchain_fn.acquire_next_image(
            self.swapchain,
            !0,
            self.acquisition_semaphores[self.acquisition_idx],
            vk::Fence::null(),
        )?;
        self.acquisition_idx = (self.acquisition_idx + 1) % self.acquisition_semaphores.len();

        Ok((image_idx as usize, acquisition_semaphore))
    }

    pub unsafe fn image(&self, idx: usize) -> Image {
        Image {
            image: self.images[idx],
            _image_memory: vk::DeviceMemory::null(),
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
        semaphores: &[vk::Semaphore],
    ) -> Result<bool, Error> {
        Ok(self.swapchain_fn.queue_present(
            self.present_queue,
            &vk::PresentInfoKHR::builder()
                .swapchains(&[self.swapchain])
                .image_indices(&[image_idx as u32])
                .wait_semaphores(semaphores)
                .build(),
        )?)
    }
}

unsafe fn choose_compute_device(
    instance: &Instance,
    devices: &[vk::PhysicalDevice],
    surface: Option<&VkSurface>,
) -> Option<(vk::PhysicalDevice, u32)> {
    for pdevice in &devices[0..] {
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
    }
}
