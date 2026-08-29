//!
//! @file lib.rs
//! @author Andrew Spaulding (Kasplat)
//! @brief Reimplements the Skyrim SE/AE versionlibdb header.
//! @bug No known bugs.
//!
//! Supports three on-disk layouts:
//!   - Format 1 (SE) / Format 2 (old AE): a length-prefixed name field followed by a
//!     sparse, delta-encoded stream of (id, offset) pairs. One entry is read per file
//!     read; there is no way to know the layout without decoding it sequentially.
//!   - Format 5 (AE 1.7.99+): a fixed 64-byte name field followed by a dense array of
//!     u32 offsets, directly indexed by id (0 meaning "no such id"). This format was
//!     introduced by Bethesda's August 2026 1.7.99 update; ported here from the
//!     reference parser in alandtse/CommonLibSSE-NG (REL/IDDB.h / IDDB.cpp).
//!

use core::ffi::CStr;
use core::fmt::Write;
use core::mem::size_of;

use cstdio::{File, Seek};

use crate::skse64::version::{SkseVersion, RUNTIME_VERSION_1_6_317};
use crate::skse64::reloc::RelocAddr;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// An item in the version database, which holds its ID and the address that maps to it.
pub struct DatabaseItem {
    pub id   : usize,
    pub addr : RelocAddr
}

/// Distinguishes the two fundamentally different body encodings we can parse.
enum DbBody {
    /// Formats 1/2: sparse, delta-encoded (id, offset) pairs. Requires sequential
    /// decode; entries are not directly indexable.
    Sparse { ptr_size: usize, prev_id: usize, prev_offset: usize },

    /// Format 5: a dense array of u32 offsets, directly indexed by id. We stream it
    /// sequentially (rather than mapping it) since we only ever need a single linear
    /// pass over it, same as the sparse case.
    Dense { next_id: usize }
}

/// A file stream for iterating over the items in a version database.
pub struct VersionDbStream {
    file              : File,
    body              : DbBody,
    remaining_entries : usize,
}

/// An enumeration used to encode how the data in an address is stored in the database.
///
/// This enumeration will be constructed directly from data read in from the database.
/// Only used by the format 1/2 (sparse) body encoding.
#[derive(Copy, Clone)]
#[repr(u8)]
#[allow(dead_code)] // Transmutes don't count as usage.
enum AddrEncoding {
    Raw64      = 0,
    Raw32      = 7,
    Raw16      = 6,
    Inc        = 1,
    PosDelta8  = 2,
    NegDelta8  = 3,
    PosDelta16 = 4,
    NegDelta16 = 5
}

// Trait used to ensure VersionDb::read only works on unsigned ints.
trait Unsigned {}
impl Unsigned for u8  {}
impl Unsigned for u16 {}
impl Unsigned for u32 {}
impl Unsigned for u64 {}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl VersionDbStream {
    /// Attempts to create a new version database, loading it with the specified version
    pub fn new(
        version: SkseVersion
    ) -> Self {
        // Large enough to hold a path for any valid game version.
        const PATH_SIZE: usize = 256;
        let mut buf = core_util::StringBuffer::<PATH_SIZE>::new();

        //
        // Note that we hard-code the build number to 0, as Bethesda doesn't use it.
        //
        // The SKSE64 team uses it to denote which store the game was obtained from, so
        // we can't just pull it from our version structure.
        //
        buf.write_fmt(format_args!(
            "Data\\SKSE\\Plugins\\{}-{}-{}-{}-0.bin",
            if version < RUNTIME_VERSION_1_6_317 { "version" } else { "versionlib" },
            version.major(),
            version.minor(),
            version.build()
        )).unwrap();

        Self::new_from_path(buf.as_c_str())
    }

    /// Creates a version database from the given path, setting the version based on the file.
    pub fn new_from_path(
        path: &CStr
    ) -> Self {
        let mut f = File::open(path, core_util::cstr!("rb")).unwrap();

        // Every format begins with a u32 tag identifying which layout follows.
        let format = Self::read::<u32>(&mut f);

        match format {
            1 | 2 => Self::new_sparse(f),
            5 => Self::new_dense(f),
            _ => panic!(
                "Unsupported address library format: {}\n\
                 This means this script extender plugin is incompatible with the address \
                 library available for this version of the game, and needs to be updated \
                 to support it.",
                format
            )
        }
    }

    ///
    /// Parses the header of a format 1/2 (sparse) version database file.
    ///
    /// The header is as follows:
    /// - (already consumed) a u32 format tag, 1 for SE or 2 for AE.
    /// - A (major, minor, build, sub) u32 tuple. This can be skipped.
    /// - A u32 module name string len, between 0 and 0x10000.
    /// - This string length is followed by exactly len many bytes encoding the name.
    /// - A u32 encoding the pointer size for the file.
    /// - A u32 count for the number of addresses in the database.
    /// - The remainder of the database is the delta-encoded addresses within it.
    ///
    fn new_sparse(
        mut f: File
    ) -> Self {
        f.seek(Seek::Current((size_of::<u32>() * 4) as i64)).unwrap();

        let mod_len = Self::read::<u32>(&mut f); // Module name length
        f.seek(Seek::Current(mod_len as i64)).unwrap();

        let (ptr_size, addr_count) = (
            Self::read::<u32>(&mut f) as usize,
            Self::read::<u32>(&mut f) as usize
        );

        Self {
            file: f,
            body: DbBody::Sparse { ptr_size, prev_id: 0, prev_offset: 0 },
            remaining_entries: addr_count
        }
    }

    ///
    /// Parses the header of a format 5 (dense, AE 1.7.99+) version database file.
    ///
    /// The header is as follows:
    /// - (already consumed) a u32 format tag, 5.
    /// - A (major, minor, build, sub) u32 tuple. This can be skipped.
    /// - A fixed-width, NUL-padded 64-byte module name field (not length-prefixed).
    /// - A u32 pointer size (unused by the dense body; read to stay aligned).
    /// - A u32 reserved "data format" field (currently unused).
    /// - A u32 count of entries in the dense array which follows.
    /// - The remainder of the database is `offset_count` many u32 offsets, directly
    ///   indexed by id. A value of 0 means "no address for this id".
    ///
    fn new_dense(
        mut f: File
    ) -> Self {
        f.seek(Seek::Current((size_of::<u32>() * 4) as i64)).unwrap(); // version[4]
        f.seek(Seek::Current(64)).unwrap(); // fixed-width name field

        let _ptr_size = Self::read::<u32>(&mut f); // Unused by the dense format.
        let _data_fmt = Self::read::<u32>(&mut f); // Reserved.
        let offset_count = Self::read::<u32>(&mut f) as usize;

        Self {
            file: f,
            body: DbBody::Dense { next_id: 0 },
            remaining_entries: offset_count
        }
    }

    /// Read T from file.
    fn read<T: Unsigned>(
        f: &mut File
    ) -> T {
        let mut b: [u8; size_of::<u64>()] = [0; size_of::<u64>()];
        assert!(f.read(b.split_at_mut(size_of::<T>()).0).unwrap() == size_of::<T>());
        // SAFETY: We only read integer types, and ensure that the buffer is the right size.
        unsafe { core::ptr::read_unaligned(b.as_ptr() as *mut T) }
    }
}

impl Iterator for VersionDbStream {
    type Item = DatabaseItem;

    fn next(
        &mut self
    ) -> Option<Self::Item> {
        match &mut self.body {
            DbBody::Sparse { ptr_size, prev_id, prev_offset } => {
                if self.remaining_entries == 0 {
                    return None;
                }
                self.remaining_entries -= 1;

                //
                // Parses an address in the version database.
                //
                // Each address seems to be encoded as follows:
                // - First, is a control byte encoding two 3-bit values denoting an item
                //   type. The msb of the control byte determines if offset calculations
                //   should use the previous offset (0) or the poffset/ptr_size (1). We
                //   call this modified offset "tpoffset".
                // - Then, the encoded data. Relative control encoding is applied to
                //   pid/tpoffset. If the high byte of the control bit was set, the
                //   resulting offset is later multiplied by pointer size (equiv, each
                //   delta is multiplied by pointer size and we can just use poffset).
                //
                let control = Self::read::<u8>(&mut self.file);
                assert!(control & 0x08 == 0);

                // SAFETY: This is the defined encoding of the control byte.
                //         The enum is sized to always be in range.
                let (id_enc, offset_enc) = unsafe {(
                    core::mem::transmute::<u8, AddrEncoding>(control & 0x07),
                    core::mem::transmute::<u8, AddrEncoding>((control >> 4) & 0x07)
                )};

                *prev_id = id_enc.read(&mut self.file, *prev_id);
                *prev_offset = if (control & 0x80) != 0 /* is the offset by pointer? */ {
                    offset_enc.read(&mut self.file, *prev_offset / *ptr_size) * *ptr_size
                } else {
                    offset_enc.read(&mut self.file, *prev_offset)
                };

                Some(DatabaseItem { id: *prev_id, addr: RelocAddr::from_offset(*prev_offset) })
            },
            DbBody::Dense { next_id } => {
                // Unlike the sparse format, entries with a zero offset are placeholders
                // ("no address for this id") and must be skipped rather than yielded, so
                // we loop until we find a populated slot or run out of entries.
                while self.remaining_entries > 0 {
                    self.remaining_entries -= 1;
                    let id = *next_id;
                    *next_id += 1;

                    let offset = Self::read::<u32>(&mut self.file) as usize;
                    if offset != 0 {
                        return Some(DatabaseItem { id, addr: RelocAddr::from_offset(offset) });
                    }
                }
                None
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl AddrEncoding {
    /// Uses an address encoding to read in new data from the file, returning the result.
    fn read(
        self,
        f: &mut File,
        prev: usize
    ) -> usize {
        match self {
            Self::Raw64      => VersionDbStream::read::<u64>(f) as usize,
            Self::Raw32      => VersionDbStream::read::<u32>(f) as usize,
            Self::Raw16      => VersionDbStream::read::<u16>(f) as usize,
            Self::Inc        => prev + 1,
            Self::PosDelta8  => prev + (VersionDbStream::read::<u8>(f) as usize),
            Self::NegDelta8  => prev - (VersionDbStream::read::<u8>(f) as usize),
            Self::PosDelta16 => prev + (VersionDbStream::read::<u16>(f) as usize),
            Self::NegDelta16 => prev - (VersionDbStream::read::<u16>(f) as usize)
        }
    }
}
