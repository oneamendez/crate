// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Sorting a guest instruction into what the translator must do with it (Arm
//! ARM, the A64 encoding index).
//!
//! Most instructions are copied into the host block unchanged, because the
//! guest's registers live in the host's registers of the same number while
//! the block runs. Only the ones that touch the machine are rewritten:
//! anything that reads or writes the program counter (branches and the
//! PC-relative addresses), anything that goes through the guest's memory
//! (loads and stores), the system registers, the barriers and cache and TLB
//! maintenance, and the instructions that change the exception level. The
//! stack pointer is virtualised too, so the few data-processing instructions
//! that name it as an operand are rewritten.

/// What to do with one guest instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// Copy the word into the host block unchanged.
    Copy,
    /// A hint with no effect on this machine (NOP, YIELD, the pointer
    /// authentication and branch-target hints, which the CPU does not offer,
    /// and the data-cache maintenance the translator does not need): emit
    /// nothing.
    Nop,
    /// A PC-relative address (ADR, ADRP): compute it from the guest PC into Rd.
    PcRel { adrp: bool, rd: u32, imm: i64 },
    /// An add or subtract immediate that names SP: `rd = rn +/- imm12<<sh`.
    AddSubSpImm {
        sub: bool,
        sf: bool,
        set_flags: bool,
        rd: u32,
        rn: u32,
        imm12: u32,
        shift12: bool,
    },
    /// AND, ORR or EOR (immediate) into SP, which is what Rd 31 means for
    /// them (with flags, ANDS, it is the zero register): `word` with its
    /// source (Rn, where 31 is the zero register) as it is.
    LogicalSpImm { word: u32 },
    /// An add or subtract of an extended register that names SP.
    AddSubSpExt {
        sub: bool,
        sf: bool,
        set_flags: bool,
        rd: u32,
        rn: u32,
        rm: u32,
        option: u32,
        amount: u32,
    },
    /// A load or store the MMU must translate.
    Memory(Mem),
    /// A direct branch: B or BL to a known target.
    Branch { link: bool, target: u64 },
    /// A conditional branch: B.cond, CBZ/CBNZ, TBZ/TBNZ. `taken` is the guest
    /// target; the fall-through is the next instruction.
    CondBranch {
        word: u32,
        kind: CondKind,
        taken: u64,
    },
    /// An indirect branch: BR, BLR or RET through a register.
    IndBranch { link: bool, ret: bool, rn: u32 },
    /// MRS or MSR of a system register.
    Sysreg { read: bool, reg: u16, rt: u32 },
    /// MSR to a PSTATE field by immediate (DAIFSet/Clr, SPSel, ...).
    MsrImm { op1: u32, op2: u32, crm: u32 },
    /// SVC, HVC or SMC.
    Call { kind: CallKind, imm: u32 },
    /// A hint whose only effect that matters is waiting (WFI, WFE).
    Wait { event: bool },
    /// ISB: a context synchronization event, where a CPU must see code that
    /// another CPU changed and invalidated.
    Isb,
    /// CLREX: clear the exclusive monitor.
    Clrex,
    /// DC ZVA: zero a 64-byte block of memory.
    DcZva { rt: u32 },
    /// A barrier or cache/TLB maintenance operation handled as a helper.
    System { word: u32 },
    /// An exception-returning instruction (ERET).
    Eret,
    /// A PC-relative literal load (LDR/LDRSW literal, of a general or a
    /// SIMD&FP register): a load from a known guest address.
    LiteralLoad {
        rt: u32,
        addr: u64,
        /// log2 of the access size in bytes.
        size: u32,
        signed: bool,
        /// A SIMD/FP destination.
        vector: bool,
    },
    /// Not understood; raise an Undefined-instruction exception.
    Undefined,
}

/// The register-level details of a load or store.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mem {
    pub word: u32,
    pub kind: MemKind,
    /// log2 of the access (or element) size in bytes.
    pub size: u32,
    /// Whether the loaded value is sign-extended.
    pub signed: bool,
    /// Whether the destination register is 64-bit (else the top 32 bits are
    /// zeroed).
    pub reg64: bool,
    pub load: bool,
    /// Base register (31 is SP here, not the zero register).
    pub rn: u32,
    /// First transfer register.
    pub rt: u32,
    /// Second transfer register for a pair.
    pub rt2: u32,
    /// The status register of a store-exclusive, or the compare/source
    /// register of a CAS or an atomic.
    pub rs: u32,
    /// The computed byte displacement for the simple addressing modes.
    pub offset: i64,
    /// Write the base back (pre/post-index).
    pub writeback: bool,
    /// Add the offset before the access (pre-index or unsigned offset).
    pub pre: bool,
    /// For a register offset: the offset register, its extend and shift.
    pub index_reg: Option<(u32, u32, u32)>,
    /// For a post-indexed SIMD structure access by register: the register
    /// added to the base afterwards.
    pub wb_reg: Option<u32>,
    /// Every byte the access touches, from its address.
    pub total: u32,
    /// The alignment the access requires, in bytes (1 for none).
    pub align: u32,
    pub acquire: bool,
    pub release: bool,
    /// LDTR/STTR and friends: checked as if at EL0.
    pub unprivileged: bool,
}

impl Mem {
    fn new(word: u32, kind: MemKind, size: u32, load: bool) -> Mem {
        Mem {
            word,
            kind,
            size,
            signed: false,
            reg64: size == 3,
            load,
            rn: (word >> 5) & 31,
            rt: word & 31,
            rt2: 31,
            rs: 31,
            offset: 0,
            writeback: false,
            pre: true,
            index_reg: None,
            wb_reg: None,
            total: 1 << size,
            align: 1,
            acquire: false,
            release: false,
            unprivileged: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemKind {
    /// A single general register.
    Gpr,
    /// A pair of general registers.
    Pair,
    /// A SIMD/FP register of 1, 2, 4, 8 or 16 bytes.
    Vec,
    /// A pair of SIMD/FP registers.
    VecPair,
    /// LD1..LD4 and ST1..ST4, of multiple structures, a single lane, or one
    /// element replicated: the host runs the same instruction on the host
    /// address.
    VecStruct,
    /// LDXR/LDAXR, or LDXP/LDAXP when `rt2` is not 31.
    LoadExclusive,
    /// STXR/STLXR, or STXP/STLXP when `rt2` is not 31; `rs` gets the status.
    StoreExclusive,
    /// LDAR, LDLAR and LDAPR.
    LoadAcquire,
    /// STLR and STLLR.
    StoreRelease,
    /// CAS: compare `rs` with memory, store `rt` if equal; `rs` gets the old.
    Cas,
    /// CASP: the same for the even/odd pairs at `rs` and `rt`.
    CasPair,
    /// An LSE read-modify-write (LDADD, LDCLR, LDEOR, LDSET, LD[SU]MAX/MIN, SWP).
    Atomic,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CondKind {
    /// B.cond: the 4-bit condition is in the word.
    Cond,
    Cbz {
        sf: bool,
        rt: u32,
    },
    Cbnz {
        sf: bool,
        rt: u32,
    },
    Tbz {
        rt: u32,
        bit: u32,
    },
    Tbnz {
        rt: u32,
        bit: u32,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallKind {
    Svc,
    Hvc,
    Smc,
    Brk,
}

fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

/// Classifies `word`, the instruction at guest address `pc`.
pub fn decode(word: u32, pc: u64) -> Op {
    let op0 = (word >> 25) & 0xf;
    match op0 {
        // Data processing -- immediate.
        0b1000 | 0b1001 => decode_data_imm(word, pc),
        // Branches, exception generating and system.
        0b1010 | 0b1011 => decode_branch_system(word, pc),
        // Loads and stores.
        0b0100 | 0b0110 | 0b1100 | 0b1110 => decode_load_store(word, pc),
        // Data processing -- register.
        0b0101 | 0b1101 => decode_data_reg(word),
        // The 0xce crypto encoding space contains SHA-512, SHA-3, SM3 and
        // SM4. ID_AA64ISAR0 exposes none of these features. A guest probing
        // them must take an undefined-instruction exception at its own PC.
        0b0111 if word >> 24 == 0xce => Op::Undefined,
        // Data processing -- SIMD and floating point.
        0b0111 | 0b1111 => Op::Copy,
        // Reserved / unallocated (op0 0b0000 is reserved; SVE is 0b0010).
        _ => Op::Undefined,
    }
}

fn decode_data_imm(word: u32, pc: u64) -> Op {
    let op = (word >> 23) & 0x7;
    match op {
        // PC-relative addressing.
        0b000 | 0b001 => {
            let rd = word & 31;
            let immlo = (word >> 29) & 3;
            let immhi = (word >> 5) & 0x7ffff;
            let adrp = word & (1 << 31) != 0;
            if adrp {
                let imm = sign_extend(((immhi << 2) | immlo) as u64, 21) << 12;
                Op::PcRel {
                    adrp: true,
                    rd,
                    imm: (pc & !0xfff).wrapping_add(imm as u64) as i64,
                }
            } else {
                let imm = sign_extend(((immhi << 2) | immlo) as u64, 21);
                Op::PcRel {
                    adrp: false,
                    rd,
                    imm: pc.wrapping_add(imm as u64) as i64,
                }
            }
        }
        // Add/subtract immediate: 31 means SP in Rn and Rd.
        0b010 => {
            let rd = word & 31;
            let rn = (word >> 5) & 31;
            let set_flags = word & (1 << 29) != 0;
            // Register 31 is SP for Rn always, and for Rd only without flags
            // (ADDS/SUBS with Rd 31 is a compare whose result is discarded, so
            // it copies unchanged).
            if rn == 31 || (rd == 31 && !set_flags) {
                Op::AddSubSpImm {
                    sub: word & (1 << 30) != 0,
                    sf: word & (1 << 31) != 0,
                    set_flags,
                    rd,
                    rn,
                    imm12: (word >> 10) & 0xfff,
                    shift12: word & (1 << 22) != 0,
                }
            } else {
                Op::Copy
            }
        }
        // Add/subtract immediate with tags (ADDG/SUBG): rare, not modelled.
        0b011 => Op::Undefined,
        // Logical immediate: Rd 31 is SP for AND, ORR and EOR (aligning the
        // stack is `and sp, xN, #-16`), the zero register for ANDS.
        0b100 if word & 31 == 31 && (word >> 29) & 3 != 0b11 => Op::LogicalSpImm { word },
        // Other logical, move wide, bitfield, extract: 31 is the zero register.
        _ => Op::Copy,
    }
}

fn decode_data_reg(word: u32) -> Op {
    // Add/subtract extended register: op bits 28:24 == 0b01011, bit 21 set,
    // option field in 15:13. 31 means SP for Rn (and Rd when not setting flags).
    if (word >> 24) & 0x1f == 0b01011 && word & (1 << 21) != 0 && (word >> 22) & 3 == 0 {
        let rd = word & 31;
        let rn = (word >> 5) & 31;
        let set_flags = word & (1 << 29) != 0;
        // Rd is SP only when flags are not set; with flags Rd 31 is XZR.
        if rn == 31 || (rd == 31 && !set_flags) {
            return Op::AddSubSpExt {
                sub: word & (1 << 30) != 0,
                sf: word & (1 << 31) != 0,
                set_flags,
                rd,
                rn,
                rm: (word >> 16) & 31,
                option: (word >> 13) & 7,
                amount: (word >> 10) & 7,
            };
        }
    }
    Op::Copy
}

fn decode_branch_system(word: u32, pc: u64) -> Op {
    let top = (word >> 29) & 0x7;
    // Unconditional branch immediate: B and BL.
    if top & 0x3 == 0 && (word >> 26) & 0x3f & 0b011111 == 0b000101 {
        let link = word & (1 << 31) != 0;
        let imm = sign_extend((word & 0x3ff_ffff) as u64, 26) * 4;
        return Op::Branch {
            link,
            target: pc.wrapping_add(imm as u64),
        };
    }
    let op = word >> 24 & 0xff;
    match op {
        // B.cond and BC.cond.
        0b0101_0100 if word & (1 << 4) == 0 => {
            let imm = sign_extend(((word >> 5) & 0x7ffff) as u64, 19) * 4;
            Op::CondBranch {
                word,
                kind: CondKind::Cond,
                taken: pc.wrapping_add(imm as u64),
            }
        }
        // Compare and branch (CBZ/CBNZ).
        0b0011_0100 | 0b1011_0100 | 0b0011_0101 | 0b1011_0101 => {
            let sf = word & (1 << 31) != 0;
            let rt = word & 31;
            let imm = sign_extend(((word >> 5) & 0x7ffff) as u64, 19) * 4;
            let taken = pc.wrapping_add(imm as u64);
            let kind = if word & (1 << 24) == 0 {
                CondKind::Cbz { sf, rt }
            } else {
                CondKind::Cbnz { sf, rt }
            };
            Op::CondBranch { word, kind, taken }
        }
        // Test and branch (TBZ/TBNZ).
        0b0011_0110 | 0b1011_0110 | 0b0011_0111 | 0b1011_0111 => {
            let rt = word & 31;
            let bit = ((word >> 31) << 5) | ((word >> 19) & 0x1f);
            let imm = sign_extend(((word >> 5) & 0x3fff) as u64, 14) * 4;
            let taken = pc.wrapping_add(imm as u64);
            let kind = if word & (1 << 24) == 0 {
                CondKind::Tbz { rt, bit }
            } else {
                CondKind::Tbnz { rt, bit }
            };
            Op::CondBranch { word, kind, taken }
        }
        // Unconditional branch register: BR, BLR, RET (and ERET, which 10:0
        // distinguishes).
        0b1101_0110 => decode_branch_reg(word),
        // Exception generation: SVC/HVC/SMC/BRK and friends.
        0b1101_0100 => decode_exception(word),
        // System: MRS/MSR, barriers, hints, cache/TLB maintenance.
        0b1101_0101 => decode_system(word),
        _ => Op::Undefined,
    }
}

fn decode_branch_reg(word: u32) -> Op {
    let opc = (word >> 21) & 0xf;
    let rn = (word >> 5) & 31;
    // op2 (20:16) must be all ones and op3 (15:10) zero for the plain forms.
    let plain = (word >> 16) & 31 == 31 && (word >> 10) & 0x3f == 0 && word & 31 == 0;
    match opc {
        0b0000 if plain => Op::IndBranch {
            link: false,
            ret: false,
            rn,
        }, // BR
        0b0001 if plain => Op::IndBranch {
            link: true,
            ret: false,
            rn,
        }, // BLR
        0b0010 if plain => Op::IndBranch {
            link: false,
            ret: true,
            rn,
        }, // RET
        0b0100 if word == 0xd69f_03e0 => Op::Eret,
        // BRAA/BRAB/RETAA/ERETAA... (pointer-authenticated branches) are not
        // offered, so they are undefined.
        _ => Op::Undefined,
    }
}

fn decode_exception(word: u32) -> Op {
    let opc = (word >> 21) & 0x7;
    let ll = word & 0x3;
    let imm = (word >> 5) & 0xffff;
    match (opc, ll) {
        (0b000, 0b01) => Op::Call {
            kind: CallKind::Svc,
            imm,
        },
        (0b000, 0b10) => Op::Call {
            kind: CallKind::Hvc,
            imm,
        },
        (0b000, 0b11) => Op::Call {
            kind: CallKind::Smc,
            imm,
        },
        (0b001, 0b00) => Op::Call {
            kind: CallKind::Brk,
            imm,
        },
        // HLT and the debug exceptions: treat as a breakpoint-like call.
        (0b010, 0b00) => Op::Call {
            kind: CallKind::Brk,
            imm,
        },
        _ => Op::Undefined,
    }
}

fn decode_system(word: u32) -> Op {
    let l = word & (1 << 21) != 0;
    let op0 = (word >> 19) & 0x3;
    let op1 = (word >> 16) & 0x7;
    let crn = (word >> 12) & 0xf;
    let crm = (word >> 8) & 0xf;
    let op2 = (word >> 5) & 0x7;
    // MSR (immediate) and hints and barriers have op0 == 0b00.
    if op0 == 0b00 {
        if crn == 0b0100 {
            // MSR (immediate) to a PSTATE field.
            return Op::MsrImm { op1, op2, crm };
        }
        if crn == 0b0010 && word & 31 == 31 {
            // Hints, numbered CRm:op2.
            return match (crm << 3) | op2 {
                2 => Op::Wait { event: true },  // WFE
                3 => Op::Wait { event: false }, // WFI
                // NOP, YIELD, SEV and SEVL (WFE does not sleep on events here,
                // see the dispatcher), DGH, ESB, CSDB, the PAC hints (the CPU
                // offers no pointer authentication, so they are NOPs, as the
                // architecture specifies), BTI and the rest.
                _ => Op::Nop,
            };
        }
        if crn == 0b0011 && word & 31 == 31 {
            return match op2 {
                0b010 => Op::Clrex,
                0b100 | 0b101 => Op::Copy, // DSB, DMB: the host's own barrier
                0b110 => Op::Isb,
                0b111 => Op::Nop, // SB, not offered
                _ => Op::Undefined,
            };
        }
        return Op::System { word };
    }
    let rt = word & 31;
    if op0 == 0b01 {
        // SYS and SYSL: cache, address-translation and TLB maintenance.
        if l {
            return Op::Undefined;
        }
        if crn == 0b0111 {
            match (op1, crm, op2) {
                (0b011, 0b0100, 0b001) => return Op::DcZva { rt },
                // DC CVAC/CVAU/CVAP/CVADP/CIVAC, and DC IVAC/ISW/CSW/CISW:
                // this machine has no caches for the guest to maintain.
                (0b011, 0b1010..=0b1110, 0b001) => return Op::Nop,
                (0b000, 0b0110 | 0b1010 | 0b1110, 0b001 | 0b010) => return Op::Nop,
                _ => {}
            }
        }
        return Op::System { word };
    }
    // MRS/MSR of a named system register (op0 0b10 or 0b11).
    let reg = sysreg_encoding(word);
    Op::Sysreg { read: l, reg, rt }
}

/// Packs a system register the way crate_core::cpu::SysReg does, from an MRS or
/// MSR encoding: op0 in bits 20:19 (plus the implicit 1), op1 18:16, CRn 15:12,
/// CRm 11:8, op2 7:5.
pub fn sysreg_encoding(word: u32) -> u16 {
    let op0 = 2 + ((word >> 19) & 1);
    let op1 = (word >> 16) & 7;
    let crn = (word >> 12) & 0xf;
    let crm = (word >> 8) & 0xf;
    let op2 = (word >> 5) & 7;
    ((op0 as u16 & 3) << 14)
        | ((op1 as u16 & 7) << 11)
        | ((crn as u16 & 15) << 7)
        | ((crm as u16 & 15) << 3)
        | (op2 as u16 & 7)
}

fn decode_load_store(word: u32, pc: u64) -> Op {
    let v = word & (1 << 26) != 0;

    // Load register (literal): bits 29:27 == 0b011, 25:24 == 0b00.
    if (word >> 27) & 0x7 == 0b011 && (word >> 24) & 0x3 == 0 {
        let opc = (word >> 30) & 0x3;
        let imm = sign_extend(((word >> 5) & 0x7ffff) as u64, 19) * 4;
        let addr = pc.wrapping_add(imm as u64);
        let (size, signed) = if v {
            if opc == 0b11 {
                return Op::Undefined;
            }
            (2 + opc, false) // S/D/Q
        } else {
            match opc {
                0b00 => (2, false),  // LDR (32-bit)
                0b01 => (3, false),  // LDR (64-bit)
                0b10 => (2, true),   // LDRSW
                _ => return Op::Nop, // PRFM (literal), a hint
            }
        };
        return Op::LiteralLoad {
            rt: word & 31,
            addr,
            size,
            signed,
            vector: v,
        };
    }

    // Load/store register pair (bits 29:27 == 0b101, 25 == 0).
    if (word >> 27) & 0x7 == 0b101 && (word >> 25) & 1 == 0 {
        return decode_pair(word, v);
    }

    // Load/store register (unscaled, immediate post/pre-index, unprivileged,
    // register offset, atomics) and unsigned immediate: bits 29:27 == 0b111.
    if (word >> 27) & 0x7 == 0b111 {
        return decode_single(word, v);
    }

    // Bits 29:27 == 0b001: with V, the SIMD structure loads and stores;
    // without, exclusive and ordered accesses and compare-and-swap.
    if (word >> 27) & 0x7 == 0b001 {
        return if v {
            decode_struct(word)
        } else {
            decode_ordered(word)
        };
    }
    Op::Undefined
}

fn decode_pair(word: u32, v: bool) -> Op {
    let opc = (word >> 30) & 0x3;
    let load = word & (1 << 22) != 0;
    let kind_bits = (word >> 23) & 0x3; // 00 no-allocate, 01 post, 10 offset, 11 pre
    let imm7 = sign_extend(((word >> 15) & 0x7f) as u64, 7);
    let (size, kind) = if v {
        if opc == 0b11 {
            return Op::Undefined;
        }
        (2 + opc, MemKind::VecPair) // 32/64/128-bit vectors
    } else {
        if opc == 0b11 || (opc == 0b01 && !load) {
            // opc 01 is LDPSW only (STGP is memory tagging, not offered).
            return Op::Undefined;
        }
        (if opc == 0b10 { 3 } else { 2 }, MemKind::Pair)
    };
    let mut m = Mem::new(word, kind, size, load);
    m.rt2 = (word >> 10) & 31;
    m.offset = imm7 << size;
    (m.pre, m.writeback) = match kind_bits {
        0b01 => (false, true), // post-index
        0b11 => (true, true),  // pre-index
        _ => (true, false),    // signed offset (and the no-allocate forms)
    };
    m.signed = !v && opc == 0b01;
    // A pair is 64-bit per element except the 32-bit non-vector form; LDPSW
    // (signed) targets 64-bit registers.
    m.reg64 = v || opc != 0b00;
    m.total = 2 << size;
    Op::Memory(m)
}

fn decode_single(word: u32, v: bool) -> Op {
    let size_field = (word >> 30) & 0x3;
    let opc = (word >> 22) & 0x3;

    // Atomic memory operations (LSE: LDADD, SWP, ...) and LDAPR share the
    // register group but set bit 21 with bits 11:10 == 0b00.
    if word & (1 << 24) == 0 && word & (1 << 21) != 0 && (word >> 10) & 3 == 0 {
        if v {
            return Op::Undefined;
        }
        let o3 = (word >> 15) & 1;
        let op = (word >> 12) & 7;
        let acquire = word & (1 << 23) != 0;
        let release = word & (1 << 22) != 0;
        let rs = (word >> 16) & 31;
        if o3 == 1 && op == 0b100 {
            // LDAPR: an acquiring load (A set, R clear, Rs all ones).
            if !acquire || release || rs != 31 {
                return Op::Undefined;
            }
            let mut m = Mem::new(word, MemKind::LoadAcquire, size_field, true);
            m.align = 1 << size_field;
            m.acquire = true;
            return Op::Memory(m);
        }
        if o3 == 1 && op != 0 {
            return Op::Undefined; // only SWP and LDAPR have o3 set
        }
        let mut m = Mem::new(word, MemKind::Atomic, size_field, true);
        m.rs = rs;
        m.align = 1 << size_field;
        m.acquire = acquire;
        m.release = release;
        return Op::Memory(m);
    }

    // PRFM (prefetch) is opc 0b10 with size 0b11 in the non-SIMD group: a hint
    // with no architectural effect.
    if !v && opc == 0b10 && size_field == 0b11 {
        return Op::Nop;
    }

    // Size and whether this is a load, its sign-extension and register width.
    let (size, load, signed, reg64) = if v {
        // SIMD: the size is (opc<1> : size_field).
        let size = ((opc & 2) << 1) | size_field;
        if size > 4 {
            return Op::Undefined;
        }
        let load = opc & 1 != 0;
        (size, load, false, false)
    } else {
        let load = opc != 0b00; // 00 STR, 01 LDR, 10/11 LDRS
        let signed = opc == 0b10 || opc == 0b11;
        if signed && size_field == 3 || (opc == 0b11 && size_field == 2) {
            return Op::Undefined;
        }
        // opc 10 sign-extends to 64 bits; opc 11 to 32. An unsigned 64-bit load
        // is size 3; everything else targets a 32-bit register.
        let reg64 = (signed && opc == 0b10) || (!signed && size_field == 3);
        (size_field, load, signed, reg64)
    };
    let kind = if v { MemKind::Vec } else { MemKind::Gpr };

    let mut m = Mem::new(word, kind, size, load);
    m.signed = signed;
    m.reg64 = reg64;

    // Unsigned immediate: bit 24 set.
    if word & (1 << 24) != 0 {
        m.offset = (((word >> 10) & 0xfff) as i64) << size;
        return Op::Memory(m);
    }

    let imm9 = sign_extend(((word >> 12) & 0x1ff) as u64, 9);
    let mode = (word >> 10) & 0x3;
    let reg_form = word & (1 << 21) != 0;
    match (mode, reg_form) {
        // Unscaled immediate (LDUR/STUR).
        (0b00, false) => m.offset = imm9,
        // Post-index.
        (0b01, false) => {
            m.offset = imm9;
            m.writeback = true;
            m.pre = false;
        }
        // Pre-index.
        (0b11, false) => {
            m.offset = imm9;
            m.writeback = true;
        }
        // Register offset: option in 15:13, S in 12.
        (0b10, true) => {
            let rm = (word >> 16) & 31;
            let option = (word >> 13) & 7;
            if option & 0b010 == 0 {
                return Op::Undefined; // only UXTW, LSL (UXTX), SXTW and SXTX
            }
            let s = (word >> 12) & 1;
            let amount = if s != 0 { size } else { 0 };
            m.index_reg = Some((rm, option, amount));
        }
        // Unprivileged load/store (LDTR/STTR family): an unscaled `[Xn,#imm9]`
        // access the kernel uses for usercopy, checked as if at EL0. There is
        // no SIMD unprivileged form.
        (0b10, false) if !v => {
            m.offset = imm9;
            m.unprivileged = true;
        }
        _ => return Op::Undefined,
    }
    Op::Memory(m)
}

/// LD1..LD4 / ST1..ST4: multiple structures, a single structure (one lane), or
/// one structure replicated to every lane (LD1R..LD4R), with optional
/// post-index writeback by the transfer size or by a register.
fn decode_struct(word: u32) -> Op {
    // Bit 31 is zero and bits 29:24 are 0b001100 (no offset or register post-
    // index forms of multiple structures) or 0b001101 (single structures).
    if word & (1 << 31) != 0 || (word >> 25) & 0x1f != 0b00110 {
        return Op::Undefined;
    }
    let q = word & (1 << 30) != 0;
    let load = word & (1 << 22) != 0;
    let post = word & (1 << 23) != 0;
    let single = word & (1 << 24) != 0;
    let rm = (word >> 16) & 31;
    let size = (word >> 10) & 3;
    let total = if !single {
        if word & (1 << 21) != 0 || (!post && rm != 0) {
            return Op::Undefined;
        }
        let (regs, selem) = match (word >> 12) & 0xf {
            0b0000 => (4, 4),
            0b0010 => (4, 1),
            0b0100 => (3, 3),
            0b0110 => (3, 1),
            0b0111 => (1, 1),
            0b1000 => (2, 2),
            0b1010 => (2, 1),
            _ => return Op::Undefined,
        };
        if size == 3 && !q && selem > 1 {
            return Op::Undefined;
        }
        regs * if q { 16 } else { 8 }
    } else {
        if !post && rm != 0 {
            return Op::Undefined;
        }
        let opcode = (word >> 13) & 7;
        let s = (word >> 12) & 1;
        let r = (word >> 21) & 1;
        let selem = (((opcode & 1) << 1) | r) + 1;
        let esize = match opcode >> 1 {
            0 => 0,
            1 if size & 1 == 0 => 1,
            2 if size == 0 => 2,
            2 if size == 1 && s == 0 => 3,
            3 if load && s == 0 => size, // replicate
            _ => return Op::Undefined,
        };
        selem << esize
    };
    let mut m = Mem::new(word, MemKind::VecStruct, 0, load);
    m.total = total;
    if post {
        m.writeback = true;
        m.pre = false;
        if rm == 31 {
            m.offset = total as i64;
        } else {
            m.wb_reg = Some(rm);
        }
    }
    Op::Memory(m)
}

/// The exclusive, acquire/release and compare-and-swap group (bits 29:24 ==
/// 0b001000).
fn decode_ordered(word: u32) -> Op {
    if (word >> 24) & 0x3f != 0b001000 {
        return Op::Undefined;
    }
    let size = (word >> 30) & 0x3;
    let o2 = word & (1 << 23) != 0;
    let l = word & (1 << 22) != 0;
    let o1 = word & (1 << 21) != 0;
    let o0 = word & (1 << 15) != 0;
    let rs = (word >> 16) & 31;
    let rt2 = (word >> 10) & 31;
    let mut m = match (o2, o1) {
        (false, false) => {
            // LDXR/LDAXR (L) or STXR/STLXR.
            let kind = if l {
                MemKind::LoadExclusive
            } else {
                MemKind::StoreExclusive
            };
            let mut m = Mem::new(word, kind, size, l);
            m.acquire = l && o0;
            m.release = !l && o0;
            m
        }
        (false, true) if size >= 2 => {
            // LDXP/LDAXP or STXP/STLXP: 32-bit (size 10) or 64-bit (11) pairs.
            let kind = if l {
                MemKind::LoadExclusive
            } else {
                MemKind::StoreExclusive
            };
            let elem = size; // 2 or 3
            let mut m = Mem::new(word, kind, elem, l);
            m.rt2 = rt2;
            m.total = 2 << elem;
            m.acquire = l && o0;
            m.release = !l && o0;
            m
        }
        (false, true) => {
            // CASP: size 00 is the 32-bit form, 01 the 64-bit one.
            let elem = size + 2;
            let mut m = Mem::new(word, MemKind::CasPair, elem, true);
            m.total = 2 << elem;
            m.acquire = l;
            m.release = o0;
            m
        }
        (true, false) => {
            // LDAR/LDLAR (L) or STLR/STLLR.
            let kind = if l {
                MemKind::LoadAcquire
            } else {
                MemKind::StoreRelease
            };
            let mut m = Mem::new(word, kind, size, l);
            m.acquire = l;
            m.release = !l;
            m
        }
        (true, true) => {
            let mut m = Mem::new(word, MemKind::Cas, size, true);
            m.acquire = l;
            m.release = o0;
            m
        }
    };
    if !matches!(m.kind, MemKind::LoadExclusive | MemKind::LoadAcquire) || o1 {
        m.rs = rs;
    }
    if m.kind == MemKind::LoadExclusive {
        m.rs = 31;
    }
    if matches!(m.kind, MemKind::LoadAcquire | MemKind::StoreRelease) {
        m.rs = 31;
    }
    m.align = m.total;
    m.reg64 = m.size == 3;
    Op::Memory(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crypto_encodings_match_the_virtual_cpu() {
        // Decode data only: never execute optional instructions on the host.
        for word in [
            0xce63c004, // SM3PARTW1 from the iPad crash report
            0xcec08420, // SM4E
            0xce628020, // SHA512H
            0xce020c20, // EOR3 (SHA-3)
        ] {
            assert_eq!(decode(word, 0x1000), Op::Undefined, "{word:#010x}");
        }
        for word in [
            0x4e284820, // AESE
            0x5e024020, // SHA256H
            0x1e604020, // FMOV
        ] {
            assert_eq!(decode(word, 0x1000), Op::Copy, "{word:#010x}");
        }
    }

    #[test]
    fn copies_ordinary_data_processing() {
        // add x0, x1, x2 ; orr x3, x4, x5 ; movz x0, #1 ; fmov d0, d1
        for w in [0x8b020020, 0xaa050083, 0xd2800020, 0x1e604020] {
            assert_eq!(decode(w, 0x1000), Op::Copy);
        }
        // add x0, x1, #4 (no SP) copies
        assert_eq!(decode(0x91001020, 0x1000), Op::Copy);
    }

    #[test]
    fn finds_sp_forms() {
        // sub sp, sp, #16
        assert!(matches!(
            decode(0xd10043ff, 0),
            Op::AddSubSpImm { rd: 31, rn: 31, .. }
        ));
        // add x29, sp, #0
        assert!(matches!(
            decode(0x910003fd, 0),
            Op::AddSubSpImm { rd: 29, rn: 31, .. }
        ));
        // and sp, x9, #0xfffffffffffffff0; orr wsp, wzr, #1; eor sp, x0, #1
        for word in [0x927cedff, 0x320003ff, 0xd240001f] {
            assert!(
                matches!(decode(word, 0), Op::LogicalSpImm { .. }),
                "{word:#x}"
            );
        }
        // ands xzr, x9, #-16 (tst) and and x0, x9, #-16 copy
        assert!(matches!(decode(0xf27ced3f, 0), Op::Copy));
        assert!(matches!(decode(0x927ced20, 0), Op::Copy));
        // add sp, sp, x0 (extended register, uxtx)
        assert!(matches!(decode(0x8b2063ff, 0), Op::AddSubSpExt { .. }));
    }

    #[test]
    fn pc_relative() {
        // adrp x0, #0x1000 ; adr x1, .+8
        assert!(matches!(
            decode(0x90000000, 0x4000),
            Op::PcRel {
                adrp: true,
                rd: 0,
                ..
            }
        ));
        match decode(0x10000041, 0x4000) {
            Op::PcRel {
                adrp: false,
                rd: 1,
                imm,
            } => assert_eq!(imm, 0x4008),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn branches() {
        // b .+8 ; bl .-4 ; ret ; br x5 ; blr x9
        assert_eq!(
            decode(0x14000002, 0x4000),
            Op::Branch {
                link: false,
                target: 0x4008
            }
        );
        assert_eq!(
            decode(0x97ffffff, 0x4000),
            Op::Branch {
                link: true,
                target: 0x3ffc
            }
        );
        assert_eq!(
            decode(0xd65f03c0, 0x4000),
            Op::IndBranch {
                link: false,
                ret: true,
                rn: 30
            }
        );
        assert!(matches!(decode(0xd61f00a0, 0), Op::IndBranch { rn: 5, .. }));
        assert!(matches!(
            decode(0xd63f0120, 0),
            Op::IndBranch {
                rn: 9,
                link: true,
                ..
            }
        ));
        // retaa is not offered
        assert_eq!(decode(0xd65f0bff, 0), Op::Undefined);
    }

    #[test]
    fn conditional_branches() {
        // b.eq .+8
        match decode(0x54000040, 0x4000) {
            Op::CondBranch {
                kind: CondKind::Cond,
                taken,
                ..
            } => assert_eq!(taken, 0x4008),
            other => panic!("{other:?}"),
        }
        // cbz x0, .+8 ; cbnz w1, .-4 ; tbz x2, #3, .+8 ; tbnz w3, #5, .+4
        assert!(matches!(
            decode(0xb4000040, 0),
            Op::CondBranch {
                kind: CondKind::Cbz { sf: true, rt: 0 },
                ..
            }
        ));
        assert!(matches!(
            decode(0x37280003, 0),
            Op::CondBranch {
                kind: CondKind::Tbnz { rt: 3, bit: 5 },
                ..
            }
        ));
    }

    #[test]
    fn system_and_exceptions() {
        // svc #0 ; hvc #0 ; brk #1 ; mrs x0, midr_el1 ; msr daifset, #2
        assert!(matches!(
            decode(0xd4000001, 0),
            Op::Call {
                kind: CallKind::Svc,
                imm: 0
            }
        ));
        assert!(matches!(
            decode(0xd4000002, 0),
            Op::Call {
                kind: CallKind::Hvc,
                ..
            }
        ));
        assert!(matches!(
            decode(0xd4200020, 0),
            Op::Call {
                kind: CallKind::Brk,
                imm: 1
            }
        ));
        match decode(0xd5380000, 0) {
            Op::Sysreg {
                read: true,
                reg,
                rt: 0,
            } => assert_eq!(reg, crate_core::cpu::SysReg::new(3, 0, 0, 0, 0).0),
            other => panic!("{other:?}"),
        }
        assert!(matches!(decode(0xd50342df, 0), Op::MsrImm { .. })); // msr daifset
        // wfi ; wfe ; nop ; paciasp ; autiasp ; bti c
        assert_eq!(decode(0xd503207f, 0), Op::Wait { event: false });
        assert_eq!(decode(0xd503205f, 0), Op::Wait { event: true });
        assert_eq!(decode(0xd503201f, 0), Op::Nop);
        assert_eq!(decode(0xd503233f, 0), Op::Nop);
        assert_eq!(decode(0xd50323bf, 0), Op::Nop);
        assert_eq!(decode(0xd503245f, 0), Op::Nop);
        // dsb ish ; isb ; clrex
        assert_eq!(decode(0xd5033bbf, 0), Op::Copy);
        assert_eq!(decode(0xd5033fdf, 0), Op::Isb);
        assert_eq!(decode(0xd5033f5f, 0), Op::Clrex);
        // dc zva, x3 ; dc cvau, x0 ; ic ivau, x0 ; tlbi vmalle1is
        assert_eq!(decode(0xd50b7423, 0), Op::DcZva { rt: 3 });
        assert_eq!(decode(0xd50b7b20, 0), Op::Nop);
        assert!(matches!(decode(0xd50b7520, 0), Op::System { .. }));
        assert!(matches!(decode(0xd508831f, 0), Op::System { .. }));
        // eret
        assert_eq!(decode(0xd69f03e0, 0), Op::Eret);
    }

    #[test]
    fn loads_and_stores() {
        // ldr x1, [x0, #8]
        match decode(0xf9400401, 0) {
            Op::Memory(m) => {
                assert!(m.load && m.size == 3 && m.offset == 8 && m.rt == 1 && m.rn == 0);
                assert_eq!(m.total, 8);
            }
            other => panic!("{other:?}"),
        }
        // str w2, [x3, #-4]! (pre-index)
        match decode(0xb81fcc62, 0) {
            Op::Memory(m) => {
                assert!(!m.load && m.writeback && m.pre && m.offset == -4 && m.size == 2);
            }
            other => panic!("{other:?}"),
        }
        // ldrb w0, [x1], #1 (post-index)
        match decode(0x38401420, 0) {
            Op::Memory(m) => {
                assert!(m.load && m.writeback && !m.pre && m.size == 0 && m.offset == 1);
            }
            other => panic!("{other:?}"),
        }
        // ldrsw x0, [x1, x2, lsl #2] (register offset, sign-extended to 64)
        match decode(0xb8a27820, 0) {
            Op::Memory(m) => {
                assert!(m.load && m.signed && m.size == 2);
                assert_eq!(m.index_reg, Some((2, 0b011, 2)));
            }
            other => panic!("{other:?}"),
        }
        // stp x0, x1, [sp, #16]
        match decode(0xa90107e0, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::Pair && m.rt == 0 && m.rt2 == 1 && m.rn == 31);
                assert_eq!(m.total, 16);
            }
            other => panic!("{other:?}"),
        }
        // prfm pldl1keep, [x0]
        assert_eq!(decode(0xf9800000, 0), Op::Nop);
    }

    #[test]
    fn ordered_and_exclusive() {
        // ldar x0, [x1]
        match decode(0xc8dffc20, 0) {
            Op::Memory(m) => assert!(m.kind == MemKind::LoadAcquire && m.load && m.align == 8),
            other => panic!("{other:?}"),
        }
        // ldaxr w2, [x3] ; stlxr w4, w5, [x6]
        match decode(0x885ffc62, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::LoadExclusive && m.acquire && m.size == 2);
                assert_eq!(m.rt2, 31);
            }
            other => panic!("{other:?}"),
        }
        match decode(0x8804fcc5, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::StoreExclusive && m.release && m.rs == 4 && m.rt == 5);
            }
            other => panic!("{other:?}"),
        }
        // ldxp x0, x1, [x2] ; stxp w3, x0, x1, [x2]
        match decode(0xc87f0440, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::LoadExclusive && m.rt2 == 1 && m.total == 16)
            }
            other => panic!("{other:?}"),
        }
        match decode(0xc8230440, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::StoreExclusive && m.rt2 == 1 && m.rs == 3);
            }
            other => panic!("{other:?}"),
        }
        // casal x0, x1, [x2] ; casp x4, x5, x6, x7, [x8]
        match decode(0xc8e0fc41, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::Cas && m.acquire && m.release && m.rs == 0 && m.rt == 1);
            }
            other => panic!("{other:?}"),
        }
        match decode(0x48247d06, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::CasPair && m.rs == 4 && m.rt == 6 && m.total == 16);
            }
            other => panic!("{other:?}"),
        }
        // ldaddal x1, x2, [x3] ; swp w4, w5, [x6] ; ldapr x0, [x1]
        match decode(0xf8e10062, 0) {
            Op::Memory(m) => assert!(m.kind == MemKind::Atomic && m.rs == 1 && m.rt == 2),
            other => panic!("{other:?}"),
        }
        match decode(0xb82480c5, 0) {
            Op::Memory(m) => assert!(m.kind == MemKind::Atomic && m.rs == 4),
            other => panic!("{other:?}"),
        }
        match decode(0xf8bfc020, 0) {
            Op::Memory(m) => assert!(m.kind == MemKind::LoadAcquire && m.acquire),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn simd_structures() {
        // ld1 {v0.16b}, [x0] ; ld1 {v0.16b-v3.16b}, [x1], #64 ; st2 {v4.4s, v5.4s}, [x2], x3
        match decode(0x4c407000, 0) {
            Op::Memory(m) => assert!(m.kind == MemKind::VecStruct && m.load && m.total == 16),
            other => panic!("{other:?}"),
        }
        match decode(0x4cdf2020, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::VecStruct && m.total == 64 && m.writeback);
                assert_eq!((m.offset, m.wb_reg), (64, None));
            }
            other => panic!("{other:?}"),
        }
        match decode(0x4c838844, 0) {
            Op::Memory(m) => {
                assert!(m.kind == MemKind::VecStruct && !m.load && m.total == 32);
                assert_eq!(m.wb_reg, Some(3));
            }
            other => panic!("{other:?}"),
        }
        // ld1 {v1.s}[1], [x2] ; ld1r {v3.8h}, [x4]
        match decode(0x0d409041, 0) {
            Op::Memory(m) => assert!(m.kind == MemKind::VecStruct && m.total == 4),
            other => panic!("{other:?}"),
        }
        match decode(0x4d40c483, 0) {
            Op::Memory(m) => assert!(m.kind == MemKind::VecStruct && m.total == 2),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn literals() {
        // ldr x1, .+8 ; ldr q2, .+16 ; prfm plil1keep, .+4
        assert!(matches!(
            decode(0x58000041, 0x1000),
            Op::LiteralLoad {
                rt: 1,
                addr: 0x1008,
                size: 3,
                vector: false,
                ..
            }
        ));
        assert!(matches!(
            decode(0x9c000082, 0x1000),
            Op::LiteralLoad {
                rt: 2,
                size: 4,
                vector: true,
                ..
            }
        ));
        assert_eq!(decode(0xd8000028, 0), Op::Nop);
    }
}
