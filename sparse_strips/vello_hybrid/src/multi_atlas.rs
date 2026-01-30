// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Multi-atlas management for `vello_hybrid`.
//!
//! This module provides support for managing multiple texture atlases, allowing for handling of
//! large numbers of images.

use alloc::vec::Vec;
use guillotiere::{AllocId, Allocation, AtlasAllocator, size2};
use thiserror::Error;

/// Manages multiple texture atlases.
pub(crate) struct MultiAtlasManager {
    /// All atlases managed by this instance.
    atlases: Vec<Atlas>,
    /// Configuration for atlas management.
    config: AtlasConfig,
    /// Round-robin counter for allocation strategy.
    round_robin_counter: usize,
}

impl MultiAtlasManager {
    /// Create a new multi-atlas manager with the given configuration.
    pub(crate) fn new(config: AtlasConfig) -> Self {
        let mut manager = Self {
            atlases: Vec::new(),
            config,
            round_robin_counter: 0,
        };

        for _ in 0..config.initial_atlas_count {
            manager
                .create_atlas()
                .expect("Failed to create initial atlas");
        }

        manager
    }

    /// Get the current configuration.
    pub(crate) fn config(&self) -> &AtlasConfig {
        &self.config
    }

    /// Create a new atlas and return its ID.
    pub(crate) fn create_atlas(&mut self) -> Result<AtlasId, AtlasError> {
        if self.atlases.len() >= self.config.max_atlases {
            return Err(AtlasError::AtlasLimitReached);
        }

        let atlas_id = AtlasId::new(self.next_atlas_id());

        let atlas = Atlas::new(atlas_id, self.config.atlas_size.0, self.config.atlas_size.1);
        self.atlases.push(atlas);

        Ok(atlas_id)
    }

    pub(crate) fn next_atlas_id(&self) -> u32 {
        u32::try_from(self.atlases.len()).unwrap()
    }

    /// Try to allocate space for an image with the given dimensions.
    pub(crate) fn try_allocate(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<AtlasAllocation, AtlasError> {
        // Check if the image is too large for any atlas
        if width > self.config.atlas_size.0 || height > self.config.atlas_size.1 {
            return Err(AtlasError::TextureTooLarge { width, height });
        }

        // Try allocation based on strategy
        match self.config.allocation_strategy {
            AllocationStrategy::FirstFit => self.allocate_first_fit(width, height),
            AllocationStrategy::BestFit => self.allocate_best_fit(width, height),
            AllocationStrategy::LeastUsed => self.allocate_least_used(width, height),
            AllocationStrategy::RoundRobin => self.allocate_round_robin(width, height),
        }
    }

    /// Allocate using first-fit strategy: try atlases in order until one has space.
    fn allocate_first_fit(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<AtlasAllocation, AtlasError> {
        for atlas in &mut self.atlases {
            if let Some(allocation) = atlas.allocate(width, height) {
                return Ok(AtlasAllocation {
                    atlas_id: atlas.id,
                    allocation,
                });
            }
        }

        // Try creating a new atlas if auto-grow is enabled
        if self.config.auto_grow {
            let atlas_id = self.create_atlas()?;
            let atlas = self.atlases.last_mut().unwrap();
            if let Some(allocation) = atlas.allocate(width, height) {
                return Ok(AtlasAllocation {
                    atlas_id,
                    allocation,
                });
            }
        }

        Err(AtlasError::NoSpaceAvailable)
    }

    /// Allocate using best-fit strategy: choose the atlas with the smallest remaining space that
    /// can fit the image.
    fn allocate_best_fit(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<AtlasAllocation, AtlasError> {
        let mut best_atlas_idx = None;
        let mut best_remaining_space = u32::MAX;

        // Find the atlas with the least remaining space that can fit the image
        for (idx, atlas) in self.atlases.iter().enumerate() {
            let stats = atlas.stats();
            let remaining_space = stats.total_area - stats.allocated_area;

            if remaining_space >= width * height && remaining_space < best_remaining_space {
                best_remaining_space = remaining_space;
                best_atlas_idx = Some(idx);
            }
        }

        if let Some(idx) = best_atlas_idx {
            let atlas = &mut self.atlases[idx];
            if let Some(allocation) = atlas.allocate(width, height) {
                return Ok(AtlasAllocation {
                    atlas_id: atlas.id,
                    allocation,
                });
            }
        }

        // Fallback to first-fit if best-fit didn't work
        self.allocate_first_fit(width, height)
    }

    /// Allocate using least-used strategy: prefer the atlas with the lowest usage percentage.
    fn allocate_least_used(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<AtlasAllocation, AtlasError> {
        let mut best_atlas_idx = None;
        let mut lowest_usage = f32::MAX;

        // Find the atlas with the lowest usage percentage
        for (idx, atlas) in self.atlases.iter().enumerate() {
            let usage = atlas.stats().usage_percentage();
            if usage < lowest_usage {
                lowest_usage = usage;
                best_atlas_idx = Some(idx);
            }
        }

        if let Some(idx) = best_atlas_idx
            && let Some(allocation) = self.atlases[idx].allocate(width, height)
        {
            let atlas_id = self.atlases[idx].id;
            return Ok(AtlasAllocation {
                atlas_id,
                allocation,
            });
        }

        // Fallback to first-fit if least-used didn't work
        self.allocate_first_fit(width, height)
    }

    /// Allocate using round-robin strategy: cycle through atlases using a round-robin counter.
    fn allocate_round_robin(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<AtlasAllocation, AtlasError> {
        if self.atlases.is_empty() {
            return self.allocate_first_fit(width, height);
        }

        let start_idx = self.round_robin_counter % self.atlases.len();

        // Try starting from the round-robin position
        for i in 0..self.atlases.len() {
            let idx = (start_idx + i) % self.atlases.len();

            if let Some(allocation) = self.atlases[idx].allocate(width, height) {
                let atlas_id = self.atlases[idx].id;
                self.round_robin_counter = (idx + 1) % self.atlases.len();
                return Ok(AtlasAllocation {
                    atlas_id,
                    allocation,
                });
            }
        }

        // Try creating a new atlas if auto-grow is enabled
        if self.config.auto_grow {
            let atlas_id = self.create_atlas()?;
            let atlas = self.atlases.last_mut().unwrap();
            if let Some(allocation) = atlas.allocate(width, height) {
                self.round_robin_counter = self.atlases.len() - 1;
                return Ok(AtlasAllocation {
                    atlas_id,
                    allocation,
                });
            }
        }

        Err(AtlasError::NoSpaceAvailable)
    }

    /// Deallocate space in the specified atlas.
    pub(crate) fn deallocate(
        &mut self,
        atlas_id: AtlasId,
        alloc_id: AllocId,
        width: u32,
        height: u32,
    ) -> Result<(), AtlasError> {
        // Since atlases only grow (never deallocate) and id is the index into the atlases vec,
        // we can do a lookup instead of a linear search
        let atlas = self
            .atlases
            .get_mut(atlas_id.0 as usize)
            .ok_or(AtlasError::AtlasNotFound(atlas_id))?;
        atlas.deallocate(alloc_id, width, height);
        Ok(())
    }

    /// Get statistics for all atlases.
    pub(crate) fn atlas_stats(&self) -> Vec<(AtlasId, &AtlasUsageStats)> {
        self.atlases
            .iter()
            .map(|atlas| (atlas.id, atlas.stats()))
            .collect()
    }

    /// Get the number of atlases.
    pub(crate) fn atlas_count(&self) -> usize {
        self.atlases.len()
    }
}

impl core::fmt::Debug for MultiAtlasManager {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MultiAtlasManager")
            .field("atlas_count", &self.atlases.len())
            .field("config", &self.config)
            .field("next_atlas_id", &self.next_atlas_id())
            .field("round_robin_counter", &self.round_robin_counter)
            .field("atlases", &self.atlases)
            .finish()
    }
}

/// Represents a single atlas in the multi-atlas system.
pub(crate) struct Atlas {
    /// Unique identifier for this atlas.
    pub id: AtlasId,
    /// Guillotiere allocator for this atlas.
    allocator: AtlasAllocator,
    /// Current usage statistics.
    stats: AtlasUsageStats,
    /// Round-robin allocation counter.
    allocation_counter: u32,
}

impl Atlas {
    /// Create a new atlas with the given ID and size.
    pub(crate) fn new(id: AtlasId, width: u32, height: u32) -> Self {
        Self {
            id,
            allocator: AtlasAllocator::new(size2(width as i32, height as i32)),
            stats: AtlasUsageStats {
                allocated_area: 0,
                total_area: width * height,
                allocated_count: 0,
            },
            allocation_counter: 0,
        }
    }

    /// Try to allocate an image in this atlas.
    pub(crate) fn allocate(&mut self, width: u32, height: u32) -> Option<Allocation> {
        if let Some(allocation) = self.allocator.allocate(size2(width as i32, height as i32)) {
            self.stats.allocated_area += width * height;
            self.stats.allocated_count += 1;
            self.allocation_counter += 1;
            Some(allocation)
        } else {
            None
        }
    }

    /// Deallocate an image from this atlas.
    pub(crate) fn deallocate(&mut self, alloc_id: AllocId, width: u32, height: u32) {
        self.allocator.deallocate(alloc_id);
        self.stats.allocated_area = self.stats.allocated_area.saturating_sub(width * height);
        self.stats.allocated_count = self.stats.allocated_count.saturating_sub(1);
    }

    /// Get current usage statistics.
    pub(crate) fn stats(&self) -> &AtlasUsageStats {
        &self.stats
    }
}

impl core::fmt::Debug for Atlas {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Atlas")
            .field("id", &self.id)
            .field("stats", &self.stats)
            .field("allocation_counter", &self.allocation_counter)
            .finish_non_exhaustive()
    }
}

/// Errors that can occur during atlas operations.
#[derive(Debug, Error)]
pub(crate) enum AtlasError {
    #[error("No space available in any atlas")]
    NoSpaceAvailable,
    #[error("Maximum number of atlases reached")]
    AtlasLimitReached,
    /// The requested texture size is too large for any atlas.
    #[error("Texture too large ({width}x{height}) for atlas")]
    TextureTooLarge {
        /// The width of the requested texture.
        width: u32,
        /// The height of the requested texture.
        height: u32,
    },
    #[error("Atlas with Id {0:?} not found")]
    AtlasNotFound(AtlasId),
}

/// Unique identifier for an atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AtlasId(pub u32);

impl AtlasId {
    /// Create a new atlas ID.
    pub(crate) fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }
}

/// Usage statistics for an atlas.
#[derive(Debug, Clone)]
pub(crate) struct AtlasUsageStats {
    /// Total allocated area in pixels.
    pub allocated_area: u32,
    /// Total available area in pixels.
    pub total_area: u32,
    /// Number of allocated images.
    pub allocated_count: u32,
}

impl AtlasUsageStats {
    /// Calculate usage percentage (0.0 to 1.0).
    pub(crate) fn usage_percentage(&self) -> f32 {
        if self.total_area == 0 {
            0.0
        } else {
            self.allocated_area as f32 / self.total_area as f32
        }
    }
}

/// Result of an atlas allocation attempt.
pub(crate) struct AtlasAllocation {
    /// The atlas where the allocation was made.
    pub atlas_id: AtlasId,
    /// The allocation details from guillotiere.
    pub allocation: Allocation,
}

/// Configuration for multiple atlas support.
#[derive(Debug, Clone, Copy)]
pub struct AtlasConfig {
    /// Initial number of atlases to create.
    pub initial_atlas_count: usize,
    /// Maximum number of atlases to create.
    pub max_atlases: usize,
    /// Size of each atlas texture.
    pub atlas_size: (u32, u32),
    /// Whether to automatically create new atlases when needed.
    pub auto_grow: bool,
    /// Strategy for allocating images across atlases.
    pub allocation_strategy: AllocationStrategy,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            // In WGPU's GLES backend, heuristics are used to decide whether a texture
            // should be treated as D2 or D2Array. However, this can cause a mismatch:
            // - when depth_or_array_layers == 1, the backend assumes the texture is D2,
            // even if it was actually created as a D2Array. This issue only occurs with the GLES backend.
            //
            // @see https://github.com/gfx-rs/wgpu/blob/61e5124eb9530d3b3865556a7da4fd320d03ddc5/wgpu-hal/src/gles/mod.rs#L470-L517
            #[cfg(all(target_arch = "wasm32", feature = "wgpu"))]
            initial_atlas_count: 2,
            #[cfg(not(all(target_arch = "wasm32", feature = "wgpu")))]
            initial_atlas_count: 1,
            max_atlases: 8,
            atlas_size: (4096, 4096),
            auto_grow: true,
            allocation_strategy: AllocationStrategy::FirstFit,
        }
    }
}

/// Strategy for allocating images across multiple atlases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationStrategy {
    /// Try atlases in order until one has space.
    #[default]
    FirstFit,
    /// Choose the atlas with the smallest remaining space that can fit the image.
    BestFit,
    /// Prefer the atlas with the lowest usage percentage.
    LeastUsed,
    /// Cycle through atlases in round-robin fashion.
    RoundRobin,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlas_creation() {
        let mut manager = MultiAtlasManager::new(AtlasConfig {
            initial_atlas_count: 0,
            ..Default::default()
        });

        let atlas_id = manager.create_atlas().unwrap();
        assert_eq!(atlas_id.as_u32(), 0);
        assert_eq!(manager.atlas_count(), 1);
    }

    #[test]
    fn test_allocation_strategies() {
        let mut manager = MultiAtlasManager::new(AtlasConfig {
            initial_atlas_count: 1,
            max_atlases: 3,
            atlas_size: (256, 256),
            allocation_strategy: AllocationStrategy::FirstFit,
            auto_grow: true,
        });

        // Should create atlas automatically
        let allocation = manager.try_allocate(100, 100).unwrap();
        assert_eq!(allocation.atlas_id.as_u32(), 0);
    }

    #[test]
    fn test_atlas_limit() {
        let mut manager = MultiAtlasManager::new(AtlasConfig {
            initial_atlas_count: 1,
            max_atlases: 1,
            atlas_size: (256, 256),
            allocation_strategy: AllocationStrategy::FirstFit,
            auto_grow: false,
        });

        assert!(manager.create_atlas().is_err());
    }

    #[test]
    fn test_texture_too_large() {
        let mut manager = MultiAtlasManager::new(AtlasConfig {
            atlas_size: (256, 256),
            ..Default::default()
        });

        let result = manager.try_allocate(300, 300);
        assert!(matches!(result, Err(AtlasError::TextureTooLarge { .. })));
    }

    #[test]
    fn test_first_fit_allocation_strategy() {
        let mut manager = MultiAtlasManager::new(AtlasConfig {
            initial_atlas_count: 3,
            max_atlases: 3,
            atlas_size: (256, 256),
            allocation_strategy: AllocationStrategy::FirstFit,
            auto_grow: false,
        });

        // First allocation should go to atlas 0
        let allocation0 = manager.try_allocate(100, 100).unwrap();
        assert_eq!(allocation0.atlas_id.as_u32(), 0);

        // Second allocation should also go to atlas 0 (first fit)
        let allocation1 = manager.try_allocate(50, 50).unwrap();
        assert_eq!(allocation1.atlas_id.as_u32(), 0);

        // Third allocation should still go to atlas 0 (first fit continues to use same atlas)
        let allocation2 = manager.try_allocate(80, 80).unwrap();
        assert_eq!(allocation2.atlas_id.as_u32(), 0);

        // Try to allocate something very large that definitely won't fit in atlas 0's remaining space
        // This should force it to go to atlas 1
        let allocation3 = manager.try_allocate(200, 200).unwrap();
        assert_eq!(allocation3.atlas_id.as_u32(), 1);

        // Next small allocation should go back to atlas 0 (first fit tries atlas 0 first)
        let allocation4 = manager.try_allocate(20, 20).unwrap();
        assert_eq!(allocation4.atlas_id.as_u32(), 0);
    }

    #[test]
    fn test_best_fit_allocation_strategy() {
        let mut manager = MultiAtlasManager::new(AtlasConfig {
            initial_atlas_count: 3,
            max_atlases: 3,
            atlas_size: (256, 256),
            allocation_strategy: AllocationStrategy::BestFit,
            auto_grow: false,
        });

        // All atlases start empty, so first allocation goes to atlas 0 (first available)
        let allocation0 = manager.try_allocate(150, 150).unwrap();
        assert_eq!(allocation0.atlas_id.as_u32(), 0);

        // Second allocation should also go to atlas 0 since it still has the least remaining space
        // that can fit the image (all atlases have same remaining space, so it picks the first)
        let allocation1 = manager.try_allocate(100, 100).unwrap();
        assert_eq!(allocation1.atlas_id.as_u32(), 0);

        // Now atlas 0 has less remaining space than atlases 1 and 2
        // For a small allocation, it should still go to atlas 0 (best fit - least remaining space)
        let allocation2 = manager.try_allocate(100, 100).unwrap();
        assert_eq!(allocation2.atlas_id.as_u32(), 0);

        // Now try to allocate something very large that won't fit in atlas 0's remaining space
        // This should force it to go to atlas 1 (which has the most remaining space)
        let allocation3 = manager.try_allocate(200, 200).unwrap();
        assert_eq!(allocation3.atlas_id.as_u32(), 1);

        // Now atlas 1 has less remaining space
        // A small allocation should go to atlas 0 as it can
        let allocation4 = manager.try_allocate(80, 80).unwrap();
        assert_eq!(allocation4.atlas_id.as_u32(), 0);

        // Now atlas 1 has less remaining space but it can't fit the allocation
        // It should go to atlas 2 (best fit - least remaining space)
        let allocation5 = manager.try_allocate(80, 80).unwrap();
        assert_eq!(allocation5.atlas_id.as_u32(), 2);
    }

    #[test]
    fn test_least_used_allocation_strategy() {
        let mut manager = MultiAtlasManager::new(AtlasConfig {
            initial_atlas_count: 3,
            max_atlases: 3,
            atlas_size: (256, 256),
            allocation_strategy: AllocationStrategy::LeastUsed,
            auto_grow: false,
        });

        // First allocation goes to atlas 0 (all atlases have 0% usage, picks first)
        let allocation0 = manager.try_allocate(100, 100).unwrap();
        assert_eq!(allocation0.atlas_id.as_u32(), 0);

        // Second allocation should go to atlas 1 (least used among remaining)
        let allocation1 = manager.try_allocate(50, 50).unwrap();
        assert_eq!(allocation1.atlas_id.as_u32(), 1);

        // Third allocation should go to atlas 2 (least used)
        let allocation2 = manager.try_allocate(30, 30).unwrap();
        assert_eq!(allocation2.atlas_id.as_u32(), 2);

        // Fourth allocation should go to atlas 2 again (still least used)
        let allocation3 = manager.try_allocate(30, 30).unwrap();
        assert_eq!(allocation3.atlas_id.as_u32(), 2);
    }

    #[test]
    fn test_round_robin_allocation_strategy() {
        let mut manager = MultiAtlasManager::new(AtlasConfig {
            initial_atlas_count: 3,
            max_atlases: 3,
            atlas_size: (256, 256),
            allocation_strategy: AllocationStrategy::RoundRobin,
            auto_grow: false,
        });

        // Allocations should cycle through atlases in order
        let allocation0 = manager.try_allocate(50, 50).unwrap();
        assert_eq!(allocation0.atlas_id.as_u32(), 0);

        let allocation1 = manager.try_allocate(50, 50).unwrap();
        assert_eq!(allocation1.atlas_id.as_u32(), 1);

        let allocation2 = manager.try_allocate(50, 50).unwrap();
        assert_eq!(allocation2.atlas_id.as_u32(), 2);

        // Should wrap back to atlas 0
        let allocation3 = manager.try_allocate(50, 50).unwrap();
        assert_eq!(allocation3.atlas_id.as_u32(), 0);

        // Continue the cycle
        let allocation4 = manager.try_allocate(50, 50).unwrap();
        assert_eq!(allocation4.atlas_id.as_u32(), 1);
    }

    #[test]
    fn test_auto_grow() {
        let mut manager = MultiAtlasManager::new(AtlasConfig {
            initial_atlas_count: 1,
            max_atlases: 3,
            atlas_size: (256, 256),
            allocation_strategy: AllocationStrategy::FirstFit,
            auto_grow: true,
        });

        let allocation0 = manager.try_allocate(256, 256).unwrap();
        assert_eq!(allocation0.atlas_id.as_u32(), 0);

        let allocation1 = manager.try_allocate(256, 256).unwrap();
        assert_eq!(allocation1.atlas_id.as_u32(), 1);

        let allocation2 = manager.try_allocate(256, 256).unwrap();
        assert_eq!(allocation2.atlas_id.as_u32(), 2);
    }
}
