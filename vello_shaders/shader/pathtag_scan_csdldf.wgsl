// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

#import config
#import pathtag

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage, read_write> reduced: array<array<atomic<u32>, PATH_MEMBERS>>;

@group(0) @binding(3)
var<storage, read_write> tag_monoids: array<array<u32, PATH_MEMBERS>>;

@group(0) @binding(4)
var<storage, read_write> scan_bump: atomic<u32>;

//Workgroup info
let LG_WG_SIZE = 8u;
let WG_SIZE = 256u;

//For the decoupled lookback
let FLAG_NOT_READY = 0u;
let FLAG_REDUCTION = 1u;
let FLAG_INCLUSIVE = 2u;
let FLAG_MASK = 3u;

//For the decoupled fallback
let MAX_SPIN_COUNT = 4u;
let LOCKED = 1u;
let UNLOCKED = 0u;

var<workgroup> sh_broadcast: u32;
var<workgroup> sh_lock: u32;
var<workgroup> sh_scratch: array<array<u32, PATH_MEMBERS>, WG_SIZE>;
var<workgroup> sh_tag_broadcast: array<u32, PATH_MEMBERS>;
var<workgroup> sh_fallback_state: array<bool, PATH_MEMBERS>;

struct pathtag_wrapper{
    p: array<u32, PATH_MEMBERS>
}

struct state_wrapper{
    s: array<bool, PATH_MEMBERS>
}

fn clear_pathtag()->array<u32, PATH_MEMBERS>{
    return array(0u, 0u, 0u, 0u, 0u);
}

fn clear_state()->array<bool, PATH_MEMBERS>{
    return array(false, false, false, false, false);
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    //acquire the partition index, set the lock
    if local_id.x == 0u {
        sh_broadcast = atomicAdd(&scan_bump, 1u);
        sh_lock = LOCKED;
    }
    workgroupBarrier();
    let part_ix = sh_broadcast;

    //Local Scan, Hillis-Steel/Kogge-Stone
    let tag_word = scene[config.pathtag_base + local_id.x + part_ix * WG_SIZE];
    var agg: pathtag_wrapper;
    agg.p = reduce_tag_arr(tag_word);
    sh_scratch[local_id.x] = agg.p;
    for (var i = 0u; i < LG_WG_SIZE; i += 1u) {
        workgroupBarrier();
        if local_id.x >= 1u << i {
            var other: pathtag_wrapper;
            other.p = sh_scratch[local_id.x - (1u << i)];
            for (var k = 0u; k < 5u; k += 1u){
                agg.p[k] += other.p[k];
            }
        }
        workgroupBarrier();
        if i < LG_WG_SIZE - 1u {
            sh_scratch[local_id.x] = agg.p;
        }
    }

    //Broadcast the results and flag into device memory
    if local_id.x == WG_SIZE - 1u {
        for (var i = 0u; i < PATH_MEMBERS; i += 1u) {
            atomicStore(&reduced[part_ix][i], (agg.p[i] << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, part_ix != 0u));
        }
    }

    //Lookback and potentially fallback
    if part_ix != 0u {
        var lookback_ix = part_ix - 1u;
        var inc_complete: state_wrapper;
        inc_complete.s = clear_state();
        var prev_reduction: pathtag_wrapper;
        prev_reduction.p = clear_pathtag();

        while(sh_lock == LOCKED){
            workgroupBarrier();

            //Lookback, with a single thread
            //Last thread in the workgroup has the complete aggregate
            if local_id.x == WG_SIZE - 1u {
                var red_complete: state_wrapper;
                red_complete.s = clear_state();
                var can_advance: bool;
                for (var spin_count = 0u; spin_count < MAX_SPIN_COUNT; ) {
                    //Attempt Lookback
                    can_advance = true;
                    for (var i = 0u; i < PATH_MEMBERS; i += 1u) {
                        if !inc_complete.s[i] && !red_complete.s[i] {
                            let payload = atomicLoad(&reduced[lookback_ix][i]);
                            let flag_value = payload & FLAG_MASK;
                            if flag_value == FLAG_REDUCTION {
                                spin_count = 0u;
                                prev_reduction.p[i] += payload >> 2u;
                                red_complete.s[i] = true;
                            } else if flag_value == FLAG_INCLUSIVE {
                                spin_count = 0u;
                                prev_reduction.p[i] += payload >> 2u;
                                atomicStore(&reduced[part_ix][i], ((agg.p[i] + prev_reduction.p[i])  << 2u) | FLAG_INCLUSIVE);
                                sh_tag_broadcast[i] = prev_reduction.p[i];
                                inc_complete.s[i] = true;
                            } else {
                                can_advance = false;
                            }
                        }
                    }

                    //Have we completed the current reduction or inclusive sum for all PathTag members?
                    if can_advance {
                        //Are all lookbacks complete?
                        var all_complete = inc_complete.s[0];
                        for (var i = 1u; i < PATH_MEMBERS; i += 1u) {
                            all_complete = all_complete && inc_complete.s[i];
                        }
                        if all_complete {
                            sh_lock = UNLOCKED;
                            break;
                        } else {
                            lookback_ix--;
                            red_complete.s = clear_state();
                        }
                    } else {
                        spin_count++;
                    }
                }

                //If we didn't complete the lookback within the allotted spins,
                //prepare for the fallback by broadcasting the lookback tile id
                //and states of the tagmonoid struct members
                if !can_advance {
                    sh_broadcast = lookback_ix;
                    for (var i = 0u; i < PATH_MEMBERS; i += 1u) {
                        sh_fallback_state[i] = !inc_complete.s[i] && !red_complete.s[i];
                    }
                }
            }
            workgroupBarrier();

            //Fallback
            if sh_lock == LOCKED {
                let fallback_ix = sh_broadcast;
                var should_fallback: state_wrapper; 
                for (var i = 0u; i < PATH_MEMBERS; i += 1u) {
                    should_fallback.s[i] = sh_fallback_state[i];
                }

                //Fallback Reduce
                //Is there an alternative to this besides a giant switch statement or
                //5 individual reductions?
                let f_word = scene[config.pathtag_base + local_id.x + fallback_ix * WG_SIZE];
                var f_agg: pathtag_wrapper;
                f_agg.p = reduce_tag_arr(f_word);
                sh_scratch[local_id.x] = f_agg.p;
                for (var i = 0u; i < LG_WG_SIZE; i += 1u) {
                    workgroupBarrier();
                    let index = i32(local_id.x) - i32(1u << i);
                    if index >= 0 {
                        for (var k = 0u; k < PATH_MEMBERS; k += 1u) {
                            if should_fallback.s[k] {
                                f_agg.p[k] += sh_scratch[index][k];
                            }
                        }
                    }
                    workgroupBarrier();
                    if i < LG_WG_SIZE - 1u {
                        for (var k = 0u; k < PATH_MEMBERS; k += 1u) {
                            if should_fallback.s[k] {
                                sh_scratch[local_id.x][k] = f_agg.p[k];
                            }
                        }
                    }
                }

                //Fallback and attempt insertion of status flag
                if local_id.x == WG_SIZE - 1u {
                    //Fallback
                    for (var i = 0u; i < PATH_MEMBERS; i += 1u) {
                        if should_fallback.s[i] {
                            let fallback_payload = (f_agg.p[i] << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_ix != 0u);
                            let prev_payload = atomicMax(&reduced[fallback_ix][i], fallback_payload);
                            if prev_payload == 0u {
                                prev_reduction.p[i] += f_agg.p[i];
                            } else {
                                prev_reduction.p[i] += prev_payload >> 2u;
                            }
                            if fallback_ix == 0u || (prev_payload & FLAG_MASK) == FLAG_INCLUSIVE {
                                atomicStore(&reduced[part_ix][i], ((agg.p[i] + prev_reduction.p[i])  << 2u) | FLAG_INCLUSIVE);
                                sh_tag_broadcast[i] = prev_reduction.p[i];
                                inc_complete.s[i] = true;
                            }
                        }
                    }

                    //At this point, the reductions are guaranteed to be complete,
                    //so try unlocking, else, keep looking back
                    var all_complete = inc_complete.s[0];
                    for (var i = 1u; i < PATH_MEMBERS; i += 1u) {
                        all_complete = all_complete && inc_complete.s[i];
                    }
                    if all_complete {
                        sh_lock = UNLOCKED;
                    } else {
                        lookback_ix--;
                    }
                }
                workgroupBarrier();
            }
        }
    }
    sh_scratch[local_id.x] = agg.p;
    workgroupBarrier();

    var tm: pathtag_wrapper;
    if part_ix != 0u {
        tm.p = sh_tag_broadcast;
    } else {
        tm.p = clear_pathtag();
    }

    if local_id.x != 0u {
        var other: pathtag_wrapper;
        other.p = sh_scratch[local_id.x - 1u];
        for (var i = 0u; i < PATH_MEMBERS; i += 1u) {
            tm.p[i] += other.p[i];
        }
    } 

    tag_monoids[local_id.x + part_ix * WG_SIZE] = tm.p;
}
