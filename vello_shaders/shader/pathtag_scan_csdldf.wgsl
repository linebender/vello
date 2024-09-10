// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

#import config
#import pathtag

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> scene: array<u32>;

@group(0) @binding(2)
var<storage, read_write> reduced: array<TagMonoidAtomic>;

@group(0) @binding(3)
var<storage, read_write> tag_monoids: array<TagMonoid>;

@group(0) @binding(4)
var<storage, read_write> scan_bump: array<atomic<u32>>;

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
var<workgroup> sh_scratch: array<TagMonoid, WG_SIZE>;
var<workgroup> sh_fallback: array<TagMonoid, WG_SIZE>;
var<workgroup> sh_tag_broadcast: TagMonoid;
var<workgroup> sh_fallback_state: array<bool, 5>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    //acquire the partition index, set the lock
    if(local_id.x == 0u){
        sh_broadcast = atomicAdd(&scan_bump[0u], 1u);
        sh_lock = LOCKED;
    }
    workgroupBarrier();
    let part_ix = sh_broadcast;

    //Local Scan, Hillis-Steel/Kogge-Stone
    let tag_word = scene[config.pathtag_base + local_id.x + part_ix * WG_SIZE];
    var agg = reduce_tag(tag_word);
    sh_scratch[local_id.x] = agg;
    for (var i = 0u; i < LG_WG_SIZE; i += 1u) {
        workgroupBarrier();
        if local_id.x >= 1u << i {
            let other = sh_scratch[local_id.x - (1u << i)];
            agg = combine_tag_monoid(other, agg);
        }
        workgroupBarrier();
        sh_scratch[local_id.x] = agg;
    }

    //Broadcast the results and results into device memory
    if local_id.x == WG_SIZE - 1u {
        if(part_ix != 0u){
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
    if(part_ix != 0u){
        var lookback_id = part_ix - 1u;
        
        var inc: array<bool, 5>;
        inc[0] = false;
        inc[1] = false;
        inc[2] = false;
        inc[3] = false;
        inc[4] = false;

        var prev: TagMonoid;
        prev[0] = 0u;
        prev[1] = 0u;
        prev[2] = 0u;
        prev[3] = 0u;
        prev[4] = 0u;

        while(sh_lock == LOCKED){
            workgroupBarrier();
            
            var red: array<bool, 5>;
            red[0] = false;
            red[1] = false;
            red[2] = false;
            red[3] = false;
            red[4] = false;

            //Lookback, with a single thread
            if(local_id.x == WG_SIZE - 1u){
                for(var spin_count: u32 = 0u; spin_count < MAX_SPIN_COUNT; ){
                    //TRANS_IX
                    if(!inc[0] && !red[0]){
                        let payload = atomicLoad(&reduced[lookback_id][0]);
                        let flag_value = payload & FLAG_MASK;
                        if(flag_value == FLAG_REDUCTION){
                            spin_count = 0u;
                            prev[0] += payload >> 2u;
                            red[0] = true;
                        } else if (flag_value == FLAG_INCLUSIVE){
                            spin_count = 0u;
                            prev[0] += payload >> 2u;
                            atomicStore(&reduced[part_ix][0], ((agg[0] + prev[0])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[0] = prev[0];
                            inc[0] = true;
                        }
                    }

                    //PATHSEG_IX
                    if(!inc[1] && !red[1]){
                        let payload = atomicLoad(&reduced[lookback_id][1]);
                        let flag_value = payload & FLAG_MASK;
                        if(flag_value == FLAG_REDUCTION){
                            spin_count = 0u;
                            prev[1] += payload >> 2u;
                            red[1] = true;
                        } else if (flag_value == FLAG_INCLUSIVE){
                            spin_count = 0u;
                            prev[1] += payload >> 2u;
                            atomicStore(&reduced[part_ix][1], ((agg[1] + prev[1])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[1] = prev[1];
                            inc[1] = true;
                        }
                    }

                    //PATHSEG_OFFSET
                    if(!inc[2] && !red[2]){
                        let payload = atomicLoad(&reduced[lookback_id][2]);
                        let flag_value = payload & FLAG_MASK;
                        if(flag_value == FLAG_REDUCTION){
                            spin_count = 0u;
                            prev[2] += payload >> 2u;
                            red[2] = true;
                        } else if (flag_value == FLAG_INCLUSIVE){
                            spin_count = 0u;
                            prev[2] += payload >> 2u;
                            atomicStore(&reduced[part_ix][2], ((agg[2] + prev[2])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[2] = prev[2];
                            inc[2] = true;
                        }
                    }

                    //STYLE_IX
                    if(!inc[3] && !red[3]){
                        let payload = atomicLoad(&reduced[lookback_id][3]);
                        let flag_value = payload & FLAG_MASK;
                        if(flag_value == FLAG_REDUCTION){
                            spin_count = 0u;
                            prev[3] += payload >> 2u;
                            red[3] = true;
                        } else if (flag_value == FLAG_INCLUSIVE){
                            spin_count = 0u;
                            prev[3] += payload >> 2u;
                            atomicStore(&reduced[part_ix][3], ((agg[3] + prev[3])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[3] = prev[3];
                            inc[3] = true;
                        }
                    }
                    
                    //PATH_IX
                    if(!inc[4] && !red[4]){
                        let payload = atomicLoad(&reduced[lookback_id][4]);
                        let flag_value = payload & FLAG_MASK;
                        if(flag_value == FLAG_REDUCTION){
                            spin_count = 0u;
                            prev[4] += payload >> 2u;
                            red[4] = true;
                        } else if (flag_value == FLAG_INCLUSIVE){
                            spin_count = 0u;
                            prev[4] += payload >> 2u;
                            atomicStore(&reduced[part_ix][4], ((agg[4] + prev[4])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[4] = prev[4];
                            inc[4] = true;
                        }
                    }

                    if((inc[0] || red[0]) && (inc[1] || red[1]) && (inc[2] || red[2]) && (inc[3] || red[3]) && (inc[4] || red[4])){
                        if(inc[0] && inc[1] && inc[2] && inc[3] && inc[4]){
                            sh_lock = UNLOCKED;
                            break;
                        } else {
                            lookback_id--;
                            red[0] = false;
                            red[1] = false;
                            red[2] = false;
                            red[3] = false;
                            red[4] = false;
                        }
                    } else {
                        spin_count++;
                    }
                }

                //If we didn't complete the lookback within the allotted spins,
                //prepare for the fallback by broadcasting the lookback tile id
                //and states of the tagmonoid struct members
                if(sh_lock == LOCKED){
                    sh_broadcast = lookback_id;
                    sh_fallback_state[0] = !inc[0] && !red[0];
                    sh_fallback_state[1] = !inc[1] && !red[1];
                    sh_fallback_state[2] = !inc[2] && !red[2];
                    sh_fallback_state[3] = !inc[3] && !red[3];
                    sh_fallback_state[4] = !inc[4] && !red[4];
                }
            }
            workgroupBarrier();

            //Fallback
            if(sh_lock == LOCKED){
                let fallback_id = sh_broadcast;

                red[0] = sh_fallback_state[0];
                red[1] = sh_fallback_state[1];
                red[2] = sh_fallback_state[2];
                red[3] = sh_fallback_state[3];
                red[4] = sh_fallback_state[4];

                //Fallback Reduce
                //Is there an alternative to this besides a giant switch statement?
                let f_word = scene[config.pathtag_base + local_id.x + fallback_id * WG_SIZE];
                var f_agg = reduce_tag(f_word);
                sh_fallback[local_id.x] = f_agg;
                for (var i = 0u; i < LG_WG_SIZE; i += 1u) {
                    workgroupBarrier();
                    if local_id.x + (1u << i) < WG_SIZE {
                        let index = local_id.x + (1u << i);
                        if(red[0]){
                            f_agg[0] += sh_fallback[index][0];
                        }
                        if(red[1]){
                            f_agg[1] += sh_fallback[index][1];
                        }
                        if(red[2]){
                            f_agg[2] += sh_fallback[index][2];
                        }
                        if(red[3]){
                            f_agg[3] += sh_fallback[index][3];
                        }
                        if(red[4]){
                            f_agg[4] += sh_fallback[index][4];
                        }
                    }
                    workgroupBarrier();
                    if(red[0]){
                        sh_fallback[local_id.x][0] = f_agg[0];
                    }

                    if(red[1]){
                        sh_fallback[local_id.x][1] = f_agg[1];
                    }

                    if(red[2]){
                        sh_fallback[local_id.x][2] = f_agg[2];
                    }

                    if(red[3]){
                        sh_fallback[local_id.x][3] = f_agg[3];
                    }
                    
                    if(red[4]){
                        sh_fallback[local_id.x][4] = f_agg[4];
                    }
                }

                //Fallback attempt insertion
                if(local_id.x == WG_SIZE - 1u){
                    //TRANS_IX FALLBACK
                    if(red[0]){
                        let fallback_payload = (f_agg[0] << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_id != 0u);
                        let prev_payload = atomicMax(&reduced[fallback_id][0], fallback_payload);
                        if(prev_payload == 0u){
                            prev[0] += f_agg[0];
                        } else {
                            prev[0] += prev_payload >> 2u;
                        }
                        if(fallback_id == 0u || (prev_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                            atomicStore(&reduced[part_ix][0], ((agg[0] + prev[0])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[0] = prev[0];
                            inc[0] = true;
                        }
                    }

                    //PATHSEG_IX FALLBACK
                    if(red[1]){
                        let fallback_payload = (f_agg[1] << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_id != 0u);
                        let prev_payload = atomicMax(&reduced[fallback_id][1], fallback_payload);
                        if(prev_payload == 0u){
                            prev[1] += f_agg[1];
                        } else {
                            prev[1] += prev_payload >> 2u;
                        }
                        if(fallback_id == 0u || (prev_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                            atomicStore(&reduced[part_ix][1], ((agg[1] + prev[1])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[1] = prev[1];
                            inc[1] = true;
                        }
                    }

                    //PATHSEG_OFFSET FALLBACK
                    if(red[2]){
                        let fallback_payload = (f_agg[2] << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_id != 0u);
                        let prev_payload = atomicMax(&reduced[fallback_id][2], fallback_payload);
                        if(prev_payload == 0u){
                            prev[2] += f_agg[2];
                        } else {
                            prev[2] += prev_payload >> 2u;
                        }
                        if(fallback_id == 0u || (prev_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                            atomicStore(&reduced[part_ix][2], ((agg[2] + prev[2])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[2] = prev[2];
                            inc[2] = true;
                        }
                    }

                    //STYLE_IX FALLBACK
                    if(red[3]){
                        let fallback_payload = (f_agg[3] << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_id != 0u);
                        let prev_payload = atomicMax(&reduced[fallback_id][3], fallback_payload);
                        if(prev_payload == 0u){
                            prev[3] += f_agg[3];
                        } else {
                            prev[3] += prev_payload >> 2u;
                        }
                        if(fallback_id == 0u || (prev_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                            atomicStore(&reduced[part_ix][3], ((agg[3] + prev[3])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[3] = prev[3];
                            inc[3] = true;
                        }
                    }

                    //PATH_IX FALLBACK
                    if(red[4]){
                        let fallback_payload = (f_agg[4] << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_id != 0u);
                        let prev_payload = atomicMax(&reduced[fallback_id][4], fallback_payload);
                        if(prev_payload == 0u){
                            prev[4] += f_agg[4];
                        } else {
                            prev[4] += prev_payload >> 2u;
                        }
                        if(fallback_id == 0u || (prev_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                            atomicStore(&reduced[part_ix][4], ((agg[4] + prev[4])  << 2u) | FLAG_INCLUSIVE);
                            sh_tag_broadcast[4] = prev[4];
                            inc[4] = true;
                        }
                    }

                    //At this point, the reductions are guaranteed to be complete,
                    //so try unlocking, else, keep looking back
                    if(inc[0] && inc[1] && inc[2] && inc[3] && inc[4]){
                        sh_lock = UNLOCKED;
                    } else {
                        lookback_id--;
                    }
                }
                workgroupBarrier();
            }
        }
    }
    workgroupBarrier();

    var tm: TagMonoid;
    if(part_ix != 0u){
        tm = sh_tag_broadcast;
    } else {
        tm[0] = 0u;
        tm[1] = 0u;
        tm[2] = 0u;
        tm[3] = 0u;
        tm[4] = 0u;
    }

    if(local_id.x != 0u){
        tm = combine_tag_monoid(tm, sh_scratch[local_id.x - 1u]);
    }

    tag_monoids[local_id.x + part_ix * WG_SIZE] = tm;
}
