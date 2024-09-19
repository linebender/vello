// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

#import config
#import pathtag

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage, read_write> reduced: array<array<atomic<u32>, 5>>;

@group(0) @binding(3)
var<storage, read_write> tag_monoids: array<array<u32, 5>>;

@group(0) @binding(4)
var<storage, read_write> scan_bump: atomic<u32>;

//Workgroup info
let LG_WG_SIZE = 8u;
let WG_SIZE = 256u;

//For the decoupled lookback
let FLAG_NOT_READY: u32 = 0;
let FLAG_REDUCTION: u32 = 1;
let FLAG_INCLUSIVE: u32 = 2;
let FLAG_MASK: u32 = 3;

//For the decoupled fallback
let MAX_SPIN_COUNT: u32 = 4;
let LOCKED: u32 = 1;
let UNLOCKED: u32 = 0;

var<workgroup> sh_broadcast: u32;
var<workgroup> sh_lock: u32;
var<workgroup> sh_scratch: array<array<u32, 5>, WG_SIZE>;
var<workgroup> sh_fallback: array<array<u32, 5>, WG_SIZE>;
var<workgroup> sh_tag_broadcast: array<u32, 5>;
var<workgroup> sh_fallback_state: array<bool, 5>;

fn attempt_lookback(
    part_ix: u32,
    lookback_ix: u32,
    member_ix: u32,
    aggregate: u32,
    spin_count: ptr<function, u32>,
    prev: ptr<function, u32>,
    reduction_complete: ptr<function, bool>,
    inclusive_complete: ptr<function, bool>
){
    let payload: u32 = atomicLoad(&reduced[lookback_ix][member_ix]);
    let flag_value: u32 = payload & FLAG_MASK;
    if flag_value == FLAG_REDUCTION {
        *spin_count = 0u;
        *prev += payload >> 2u;
        *reduction_complete = true;
    } else if flag_value == FLAG_INCLUSIVE {
        *spin_count = 0u;
        *prev += payload >> 2u;
        atomicStore(&reduced[part_ix][member_ix], ((aggregate + *prev)  << 2u) | FLAG_INCLUSIVE);
        sh_tag_broadcast[member_ix] = *prev;
        *inclusive_complete = true;
    }
}

fn fallback(
    part_ix: u32,
    fallback_ix: u32,
    member_ix: u32,
    aggregate: u32,
    fallback_aggregate: u32,
    prev: ptr<function, u32>,
    inclusive_complete: ptr<function, bool>
){
    let fallback_payload = (fallback_aggregate << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_ix != 0u);
    let prev_payload = atomicMax(&reduced[fallback_ix][member_ix], fallback_payload);
    if prev_payload == 0u {
        *prev += fallback_aggregate;
    } else {
        *prev += prev_payload >> 2u;
    }
    if fallback_ix == 0u || (prev_payload & FLAG_MASK) == FLAG_INCLUSIVE {
        atomicStore(&reduced[part_ix][member_ix], ((aggregate + *prev)  << 2u) | FLAG_INCLUSIVE);
        sh_tag_broadcast[member_ix] = *prev;
        *inclusive_complete = true;
    }
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
    let part_ix: u32 = sh_broadcast;

    //Local Scan, Hillis-Steel/Kogge-Stone
    let tag_word: u32 = scene[config.pathtag_base + local_id.x + part_ix * WG_SIZE];
    var agg: array<u32, 5> = reduce_tag_arr(tag_word);
    sh_scratch[local_id.x] = agg;
    for (var i: u32 = 0u; i < LG_WG_SIZE; i += 1u) {
        workgroupBarrier();
        if local_id.x >= 1u << i {
            let other: array<u32, 5> = sh_scratch[local_id.x - (1u << i)];
            agg[0] += other[0];
            agg[1] += other[1];
            agg[2] += other[2];
            agg[3] += other[3];
            agg[4] += other[4];
        }
        workgroupBarrier();
        sh_scratch[local_id.x] = agg;
    }

    //Broadcast the results and flag into device memory
    if local_id.x == WG_SIZE - 1u {
        if part_ix != 0u {
            atomicStore(&reduced[part_ix][0], (agg[0] << 2u) | FLAG_REDUCTION);
            atomicStore(&reduced[part_ix][1], (agg[1] << 2u) | FLAG_REDUCTION);
            atomicStore(&reduced[part_ix][2], (agg[2] << 2u) | FLAG_REDUCTION);
            atomicStore(&reduced[part_ix][3], (agg[3] << 2u) | FLAG_REDUCTION);
            atomicStore(&reduced[part_ix][4], (agg[4] << 2u) | FLAG_REDUCTION);
        } else {
            atomicStore(&reduced[part_ix][0], (agg[0] << 2u) | FLAG_INCLUSIVE);
            atomicStore(&reduced[part_ix][1], (agg[1] << 2u) | FLAG_INCLUSIVE);
            atomicStore(&reduced[part_ix][2], (agg[2] << 2u) | FLAG_INCLUSIVE);
            atomicStore(&reduced[part_ix][3], (agg[3] << 2u) | FLAG_INCLUSIVE);
            atomicStore(&reduced[part_ix][4], (agg[4] << 2u) | FLAG_INCLUSIVE);
        }
    }

    //Lookback and potentially fallback
    if part_ix != 0u {
        var lookback_ix = part_ix - 1u;
        
        var inc0: bool = false;
        var inc1: bool = false;
        var inc2: bool = false;
        var inc3: bool = false;
        var inc4: bool = false;

        var prev0: u32 = 0u;
        var prev1: u32 = 0u;
        var prev2: u32 = 0u;
        var prev3: u32 = 0u;
        var prev4: u32 = 0u;

        while(sh_lock == LOCKED){
            workgroupBarrier();
            
            var red0: bool = false;
            var red1: bool = false;
            var red2: bool = false;
            var red3: bool = false;
            var red4: bool = false;
            
            //Lookback, with a single thread
            //Last thread in the workgroup has the complete aggregate
            if local_id.x == WG_SIZE - 1u {
                for (var spin_count: u32 = 0u; spin_count < MAX_SPIN_COUNT; ) {
                    //TRANS_IX
                    if !inc0 && !red0 {
                        attempt_lookback(
                            part_ix,
                            lookback_ix,
                            0u,
                            agg[0],
                            &spin_count,
                            &prev0,
                            &red0,
                            &inc0);
                    }

                    //PATHSEG_IX
                    if !inc1 && !red1 {
                        attempt_lookback(
                            part_ix,
                            lookback_ix,
                            1u,
                            agg[1],
                            &spin_count,
                            &prev1,
                            &red1,
                            &inc1);
                    }

                    //PATHSEG_OFFSET
                    if !inc2 && !red2 {
                        attempt_lookback(
                            part_ix,
                            lookback_ix,
                            2u,
                            agg[2],
                            &spin_count,
                            &prev2,
                            &red2,
                            &inc2);
                    }

                    //STYLE_IX
                    if !inc3 && !red3 {
                        attempt_lookback(
                            part_ix,
                            lookback_ix,
                            3u,
                            agg[3],
                            &spin_count,
                            &prev3,
                            &red3,
                            &inc3);
                    }
                    
                    //PATH_IX
                    if !inc4 && !red4 {
                        attempt_lookback(
                            part_ix,
                            lookback_ix,
                            4u,
                            agg[4],
                            &spin_count,
                            &prev4,
                            &red4,
                            &inc4);
                    }

                    //Have we completed the current reduction or inclusive sum for all PathTag members?
                    if (inc0 || red0) && (inc1 || red1) && (inc2 || red2) && (inc3 || red3) && (inc4 || red4) {
                        if inc0 && inc1 && inc2 && inc3 && inc4 {
                            sh_lock = UNLOCKED;
                            break;
                        } else {
                            lookback_ix--;
                            red0 = false;
                            red1 = false;
                            red2 = false;
                            red3 = false;
                            red4 = false;
                        }
                    } else {
                        spin_count++;
                    }
                }

                //If we didn't complete the lookback within the allotted spins,
                //prepare for the fallback by broadcasting the lookback tile id
                //and states of the tagmonoid struct members
                if sh_lock == LOCKED {
                    sh_broadcast = lookback_ix;
                    sh_fallback_state[0] = !inc0 && !red0;
                    sh_fallback_state[1] = !inc1 && !red1;
                    sh_fallback_state[2] = !inc2 && !red2;
                    sh_fallback_state[3] = !inc3 && !red3;
                    sh_fallback_state[4] = !inc4 && !red4;
                }
            }
            workgroupBarrier();

            //Fallback
            if sh_lock == LOCKED {
                let fallback_ix = sh_broadcast;

                red0 = sh_fallback_state[0];
                red1 = sh_fallback_state[1];
                red2 = sh_fallback_state[2];
                red3 = sh_fallback_state[3];
                red4 = sh_fallback_state[4];

                //Fallback Reduce
                //Is there an alternative to this besides a giant switch statement or
                //5 individual reductions?
                let f_word: u32 = scene[config.pathtag_base + local_id.x + fallback_ix * WG_SIZE];
                var f_agg: array<u32, 5> = reduce_tag_arr(f_word);
                sh_fallback[local_id.x] = f_agg;
                for (var i = 0u; i < LG_WG_SIZE; i += 1u) {
                    workgroupBarrier();
                    if local_id.x + (1u << i) < WG_SIZE {
                        let index = local_id.x + (1u << i);
                        if red0 {
                            f_agg[0] += sh_fallback[index][0];
                        }
                        if red1 {
                            f_agg[1] += sh_fallback[index][1];
                        }
                        if red2 {
                            f_agg[2] += sh_fallback[index][2];
                        }
                        if red3 {
                            f_agg[3] += sh_fallback[index][3];
                        }
                        if red4 {
                            f_agg[4] += sh_fallback[index][4];
                        }
                    }
                    workgroupBarrier();
                    if red0 {
                        sh_fallback[local_id.x][0] = f_agg[0];
                    }

                    if red1 {
                        sh_fallback[local_id.x][1] = f_agg[1];
                    }

                    if red2 {
                        sh_fallback[local_id.x][2] = f_agg[2];
                    }

                    if red3 {
                        sh_fallback[local_id.x][3] = f_agg[3];
                    }
                    
                    if red4 {
                        sh_fallback[local_id.x][4] = f_agg[4];
                    }
                }

                //Fallback and attempt insertion of status flag
                if local_id.x == WG_SIZE - 1u {
                    //TRANS_IX FALLBACK
                    if red0 {
                        fallback(
                            part_ix,
                            fallback_ix,
                            0u,
                            agg[0],
                            f_agg[0],
                            &prev0,
                            &inc0,
                        );
                    }

                    //PATHSEG_IX FALLBACK
                    if red1 {
                        fallback(
                            part_ix,
                            fallback_ix,
                            1u,
                            agg[1],
                            f_agg[1],
                            &prev1,
                            &inc1,
                        );
                    }

                    //PATHSEG_OFFSET FALLBACK
                    if red2 {
                        fallback(
                            part_ix,
                            fallback_ix,
                            2u,
                            agg[2],
                            f_agg[2],
                            &prev2,
                            &inc2,
                        );
                    }

                    //STYLE_IX FALLBACK
                    if red3 {
                        fallback(
                            part_ix,
                            fallback_ix,
                            3u,
                            agg[3],
                            f_agg[3],
                            &prev3,
                            &inc3,
                        );
                    }

                    //PATH_IX FALLBACK
                    if red4 {
                        fallback(
                            part_ix,
                            fallback_ix,
                            4u,
                            agg[4],
                            f_agg[4],
                            &prev4,
                            &inc4,
                        );
                    }

                    //At this point, the reductions are guaranteed to be complete,
                    //so try unlocking, else, keep looking back
                    if inc0 && inc1 && inc2 && inc3 && inc4 {
                        sh_lock = UNLOCKED;
                    } else {
                        lookback_ix--;
                    }
                }
                workgroupBarrier();
            }
        }
    }
    workgroupBarrier();

    var tm: array<u32, 5>;
    if part_ix != 0u {
        tm = sh_tag_broadcast;
    } else {
        tm[0] = 0u;
        tm[1] = 0u;
        tm[2] = 0u;
        tm[3] = 0u;
        tm[4] = 0u;
    }

    if local_id.x != 0u {
        let other: array<u32, 5> = sh_scratch[local_id.x - 1u];
        tm[0] += other[0];
        tm[1] += other[1];
        tm[2] += other[2];
        tm[3] += other[3];
        tm[4] += other[4]; 
    } 

    tag_monoids[local_id.x + part_ix * WG_SIZE] = tm;
}
