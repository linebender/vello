//! Vulkan implemenation of HAL trait.

use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::sync::Arc;

use ash::extensions::ext::DebugUtils;
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, Device, Entry, Instance};
use once_cell::sync::Lazy;

use crate::Error;

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
    device_mem_props: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    qfi: u32,
    timestamp_period: f32,
}

struct RawDevice {
    device: Device,
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

#[cfg(debug_assertions)]
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
        message_severity,
        message_type,
        message_id_name,
        message_id_number,
        message,
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
    pub fn new() -> Result<VkInstance, Error> {
        unsafe {
            let app_name = CString::new("VkToy").unwrap();
            let entry = Entry::new()?;

            let exist_layers = entry
                .enumerate_instance_layer_properties()?;
            let layers = LAYERS.iter().filter_map(|&lyr| {
                exist_layers
                    .iter()
                    .find(|x|
                        CStr::from_ptr(x.layer_name.as_ptr()) == lyr
                    )
                    .map(|_| lyr.as_ptr())
                    .or_else(|| {
                        println!("Unable to find layer: {}, have you installed the Vulkan SDK?", lyr.to_string_lossy());
                        None
                    })
            }).collect::<Vec<_>>();

            let exist_exts = entry
                .enumerate_instance_extension_properties()?;
            let exts = EXTS.iter().filter_map(|&ext| {
                exist_exts
                    .iter()
                    .find(|x|
                        CStr::from_ptr(x.extension_name.as_ptr()) == ext
                    )
                    .map(|_| ext.as_ptr())
                    .or_else(|| {
                        println!("Unable to find extension: {}, have you installed the Vulkan SDK?", ext.to_string_lossy());
                        None
                    })
            }).collect::<Vec<_>>();

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

            Ok(VkInstance {
                entry,
                instance,
                _dbg_loader,
                _dbg_callbk,
            })
        }
    }

    /// Create a device from the instance, suitable for compute.
    ///
    /// # Safety
    ///
    /// The caller is responsible for making sure that the instance outlives the device.
    /// We could enforce that, for example having an `Arc` of the raw instance, but for
    /// now keep things simple.
    pub unsafe fn device(&self) -> Result<VkDevice, Error> {
        let devices = self.instance.enumerate_physical_devices()?;
        let (pdevice, qfi) =
            choose_compute_device(&self.instance, &devices).ok_or("no suitable device")?;

        let device = self.instance.create_device(
            pdevice,
            &vk::DeviceCreateInfo::builder().queue_create_infos(&[
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(qfi)
                    .queue_priorities(&[1.0])
                    .build(),
            ]),
            None,
        )?;

        let device_mem_props = self.instance.get_physical_device_memory_properties(pdevice);

        let queue_index = 0;
        let queue = device.get_device_queue(qfi, queue_index);

        let device = Arc::new(RawDevice { device });

        let props = self.instance.get_physical_device_properties(pdevice);
        let timestamp_period = props.limits.timestamp_period;

        Ok(VkDevice {
            device,
            device_mem_props,
            qfi,
            queue,
            timestamp_period,
        })
    }
}

impl crate::Device for VkDevice {
    type Buffer = Buffer;
    type CmdBuf = CmdBuf;
    type DescriptorSet = DescriptorSet;
    type Pipeline = Pipeline;
    type QueryPool = QueryPool;
    type MemFlags = MemFlags;

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

    /// This creates a pipeline that runs over the buffer.
    ///
    /// The descriptor set layout is just some number of buffers (this will change).
    unsafe fn create_simple_compute_pipeline(
        &self,
        code: &[u8],
        n_buffers: u32,
    ) -> Result<Pipeline, Error> {
        let device = &self.device.device;
        let bindings = (0..n_buffers)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build()
            })
            .collect::<Vec<_>>();
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
    ) -> Result<DescriptorSet, Error> {
        let device = &self.device.device;
        let descriptor_pool_sizes = [vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(bufs.len() as u32)
            .build()];
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
        Ok(DescriptorSet {
            descriptor_set: descriptor_sets[0],
        })
    }

    fn create_cmd_buf(&self) -> Result<CmdBuf, Error> {
        unsafe {
            let device = &self.device.device;
            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::empty())
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

    unsafe fn reap_query_pool(&self, pool: Self::QueryPool) -> Result<Vec<f64>, Error> {
        let device = &self.device.device;
        let mut buf = vec![0u64; pool.n_queries as usize];
        device.get_query_pool_results(
            pool.pool,
            0,
            pool.n_queries,
            &mut buf,
            vk::QueryResultFlags::TYPE_64,
        )?;
        device.destroy_query_pool(pool.pool, None);
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
    unsafe fn run_cmd_buf(&self, cmd_buf: &CmdBuf) -> Result<(), Error> {
        let device = &self.device.device;

        // Run the command buffer.
        let fence = device.create_fence(
            &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::empty()),
            None,
        )?;
        device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::builder()
                .command_buffers(&[cmd_buf.cmd_buf])
                .build()],
            fence,
        )?;
        device.wait_for_fences(&[fence], true, 100_000_000)?;
        // TODO: handle errors better (currently leaks fence and can lead to other problems)
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

    unsafe fn reset_query_pool(&mut self, pool: &QueryPool) {
        let device = &self.device.device;
        device.cmd_reset_query_pool(
            self.cmd_buf,
            pool.pool,
            0,
            pool.n_queries,
        );
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

unsafe fn choose_compute_device(
    instance: &Instance,
    devices: &[vk::PhysicalDevice],
) -> Option<(vk::PhysicalDevice, u32)> {
    for pdevice in devices {
        let props = instance.get_physical_device_queue_family_properties(*pdevice);
        for (ix, info) in props.iter().enumerate() {
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
