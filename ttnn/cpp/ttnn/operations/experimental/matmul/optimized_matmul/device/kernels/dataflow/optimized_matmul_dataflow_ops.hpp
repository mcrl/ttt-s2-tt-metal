// Common dataflow operations for TT-Metal kernels
// Shared between baseline and optim implementations
#pragma once

#include <cstdint>

// Legacy Wormhole virtual-coordinate offset retained for callers that still
// access NoC virtual coordinates directly.
constexpr uint32_t WH_LOGICAL_TO_VIRTUAL_OFFSET = 18;

FORCE_INLINE uint32_t worker_logical_x() {
    return static_cast<uint32_t>(get_absolute_logical_x());
}

FORCE_INLINE uint32_t worker_logical_y() {
    return static_cast<uint32_t>(get_absolute_logical_y());
}

FORCE_INLINE coord_t worker_virtual_coord(const uint32_t logical_x,
                                          const uint32_t logical_y) {
    return get_virtual_coord_from_worker_logical_coord(
        static_cast<uint8_t>(logical_x), static_cast<uint8_t>(logical_y));
}

// Async read with transaction ID for fine-grained barrier control
// Use noc_async_read_barrier_with_trid(trid, noc) to wait for specific trid
// Also compatible with noc_async_read_barrier(noc) for all reads
//
// 1. Set state (coordinate + size) via noc_async_read_one_packet_set_state
// 2. Set transaction ID via ncrisc_noc_set_transaction_id
// 3. Issue read via ncrisc_noc_fast_read_with_transaction_id
FORCE_INLINE void noc_async_read_with_trid(uint64_t src_noc_addr,
                                           uint32_t dst_local_l1_addr,
                                           uint32_t size, uint32_t trid,
                                           uint8_t noc = noc_index) {
    // Step 1: Set state (coordinate + size)
    noc_async_read_one_packet_set_state(src_noc_addr, size, 0, noc);

    // Step 2: Set transaction ID
    ncrisc_noc_set_transaction_id(noc, read_cmd_buf, trid);

    // Step 3: Issue read (handles backpressure and counter increment)
    ncrisc_noc_fast_read_with_transaction_id<noc_mode, false>(
        noc, read_cmd_buf, 0, (uint32_t)src_noc_addr, dst_local_l1_addr, trid);
}

// Generic broadcast function for rectangular (including 1D) multicast
// Broadcasts L1 data from sender to a rectangular region
// Automatically uses loopback if sender is in destination range
FORCE_INLINE void broadcast(uint32_t l1_addr, uint32_t nbytes,
                            uint32_t sender_x, uint32_t sender_y, uint32_t x0,
                            uint32_t y0, uint32_t x1, uint32_t y1,
                            uint32_t sender_sem_id, uint32_t receiver_sem_id,
                            uint8_t noc) {
    // Assert valid coordinate ranges
    ASSERT(x0 <= x1);
    ASSERT(y0 <= y1);

    // Translate semaphore IDs to addresses and pointers
    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t *sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t *receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(receiver_sem_addr);

    // Get current core's logical coordinates
    const uint32_t x = worker_logical_x();
    const uint32_t y = worker_logical_y();

    const auto sender_virtual = worker_virtual_coord(sender_x, sender_y);
    const auto x0_virtual = worker_virtual_coord(x0, y0);
    const auto x1_virtual = worker_virtual_coord(x1, y1);

    // Check if sender is in destination range
    bool sender_in_range =
        (sender_x >= x0 && sender_x <= x1 && sender_y >= y0 && sender_y <= y1);

    uint32_t num_dests = (x1 - x0 + 1) * (y1 - y0 + 1);
    if (sender_in_range && num_dests == 1) {
        return;
    }

    if (x == sender_x && y == sender_y) {
        // SENDER: Wait for all receivers to signal ready
        noc_semaphore_wait(sender_sem_ptr,
                           sender_in_range ? num_dests - 1 : num_dests);
        // Reset sender semaphore to 0 for next use
        noc_semaphore_set(sender_sem_ptr, 0);

        // Multicast data
        uint64_t mcast_data_addr =
            noc == 0
                ? get_noc_multicast_addr(x0_virtual.x, x0_virtual.y,
                                         x1_virtual.x, x1_virtual.y, l1_addr,
                                         noc)
                : get_noc_multicast_addr(x1_virtual.x, x1_virtual.y,
                                         x0_virtual.x, x0_virtual.y, l1_addr,
                                         noc);
        if (sender_in_range) {
            noc_async_write_multicast_loopback_src(
                l1_addr, mcast_data_addr, nbytes, num_dests, false, noc);
        } else {
            noc_async_write_multicast(l1_addr, mcast_data_addr, nbytes,
                                      num_dests, false, noc);
        }

#ifdef ARCH_BLACKHOLE
        noc_async_writes_flushed(noc);
#endif

        // Multicast VALID signal to all receivers
        *receiver_sem_ptr = VALID;
        uint64_t mcast_signal_addr =
            noc == 0
                ? get_noc_multicast_addr(x0_virtual.x, x0_virtual.y,
                                         x1_virtual.x, x1_virtual.y,
                                         receiver_sem_addr, noc)
                : get_noc_multicast_addr(x1_virtual.x, x1_virtual.y,
                                         x0_virtual.x, x0_virtual.y,
                                         receiver_sem_addr, noc);
        if (sender_in_range) {
            noc_semaphore_set_multicast_loopback_src(
                receiver_sem_addr, mcast_signal_addr, num_dests, false, noc);
        } else {
            noc_semaphore_set_multicast(receiver_sem_addr, mcast_signal_addr,
                                        num_dests, false, noc);
        }
        noc_async_write_barrier(noc);
    } else if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
        // RECEIVER: This core is in destination range
        noc_semaphore_set(receiver_sem_ptr, INVALID);

        // Signal sender that we're ready
        uint64_t sender_sem_noc_addr = get_noc_addr(
            sender_virtual.x, sender_virtual.y, sender_sem_addr, noc);
        noc_semaphore_inc(sender_sem_noc_addr, 1, noc);

        // Wait for sender to signal data is ready
        noc_semaphore_wait(receiver_sem_ptr, VALID);
    }
    // Else: Core is neither sender nor receiver - do nothing
}

FORCE_INLINE void broadcast_v2(uint32_t l1_addr, uint32_t nbytes,
                               uint32_t sender_x, uint32_t sender_y,
                               uint32_t x0, uint32_t y0, uint32_t x1,
                               uint32_t y1, uint32_t sender_sem_id,
                               uint32_t receiver_sem_id, uint8_t noc) {
    ASSERT(x0 <= x1);
    ASSERT(y0 <= y1);

    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t *sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t *receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(receiver_sem_addr);

    const uint32_t x = worker_logical_x();
    const uint32_t y = worker_logical_y();

    const auto sender_virtual = worker_virtual_coord(sender_x, sender_y);
    const auto x0_virtual = worker_virtual_coord(x0, y0);
    const auto x1_virtual = worker_virtual_coord(x1, y1);

    bool sender_in_range =
        (sender_x >= x0 && sender_x <= x1 && sender_y >= y0 && sender_y <= y1);
    uint32_t num_dests = (x1 - x0 + 1) * (y1 - y0 + 1);
    if (sender_in_range && num_dests == 1) {
        return;
    }

    if (x == sender_x && y == sender_y) {
        noc_semaphore_wait(sender_sem_ptr,
                           sender_in_range ? num_dests - 1 : num_dests);
        noc_semaphore_set(sender_sem_ptr, 0);

        uint64_t mcast_data_addr =
            noc == 0
                ? get_noc_multicast_addr(x0_virtual.x, x0_virtual.y,
                                         x1_virtual.x, x1_virtual.y, l1_addr,
                                         noc)
                : get_noc_multicast_addr(x1_virtual.x, x1_virtual.y,
                                         x0_virtual.x, x0_virtual.y, l1_addr,
                                         noc);
        if (sender_in_range) {
            noc_async_write_multicast_loopback_src(
                l1_addr, mcast_data_addr, nbytes, num_dests, true, noc);
        } else {
            noc_async_write_multicast(l1_addr, mcast_data_addr, nbytes,
                                      num_dests, true, noc);
        }

#ifdef ARCH_BLACKHOLE
        noc_async_writes_flushed(noc);
#endif

        *receiver_sem_ptr = VALID;
        uint64_t mcast_signal_addr =
            noc == 0
                ? get_noc_multicast_addr(x0_virtual.x, x0_virtual.y,
                                         x1_virtual.x, x1_virtual.y,
                                         receiver_sem_addr, noc)
                : get_noc_multicast_addr(x1_virtual.x, x1_virtual.y,
                                         x0_virtual.x, x0_virtual.y,
                                         receiver_sem_addr, noc);
        if (sender_in_range) {
            noc_semaphore_set_multicast_loopback_src(
                receiver_sem_addr, mcast_signal_addr, num_dests, false, noc);
        } else {
            noc_semaphore_set_multicast(receiver_sem_addr, mcast_signal_addr,
                                        num_dests, false, noc);
        }
    } else if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
        noc_semaphore_set(receiver_sem_ptr, INVALID);
        uint64_t sender_sem_noc_addr = get_noc_addr(
            sender_virtual.x, sender_virtual.y, sender_sem_addr, noc);
        noc_semaphore_inc(sender_sem_noc_addr, 1, noc);
        noc_semaphore_wait(receiver_sem_ptr, VALID);
    }
}

// Async version of broadcast - initiates transfer without waiting for
// completion SENDER: Waits for receivers ready, initiates multicast, returns
// immediately RECEIVER: Signals ready, returns immediately (does NOT wait for
// data) Must call broadcast_wait() to ensure completion before using data
FORCE_INLINE void broadcast_async(uint32_t l1_addr, uint32_t nbytes,
                                  uint32_t sender_x, uint32_t sender_y,
                                  uint32_t x0, uint32_t y0, uint32_t x1,
                                  uint32_t y1, uint32_t sender_sem_id,
                                  uint32_t receiver_sem_id, uint8_t noc) {
    ASSERT(x0 <= x1);
    ASSERT(y0 <= y1);

    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t *sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t *receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(receiver_sem_addr);

    const uint32_t x = worker_logical_x();
    const uint32_t y = worker_logical_y();

    const auto sender_virtual = worker_virtual_coord(sender_x, sender_y);
    const auto x0_virtual = worker_virtual_coord(x0, y0);
    const auto x1_virtual = worker_virtual_coord(x1, y1);

    bool sender_in_range =
        (sender_x >= x0 && sender_x <= x1 && sender_y >= y0 && sender_y <= y1);
    uint32_t num_dests = (x1 - x0 + 1) * (y1 - y0 + 1);
    if (sender_in_range && num_dests == 1) {
        return;
    }

    if (x == sender_x && y == sender_y) {
        // SENDER: Wait for all receivers to signal ready
        noc_semaphore_wait(sender_sem_ptr,
                           sender_in_range ? num_dests - 1 : num_dests);
        noc_semaphore_set(sender_sem_ptr, 0);

        // Multicast data (async - no barrier)
        uint64_t mcast_data_addr =
            noc == 0
                ? get_noc_multicast_addr(x0_virtual.x, x0_virtual.y,
                                         x1_virtual.x, x1_virtual.y, l1_addr,
                                         noc)
                : get_noc_multicast_addr(x1_virtual.x, x1_virtual.y,
                                         x0_virtual.x, x0_virtual.y, l1_addr,
                                         noc);
        if (sender_in_range) {
            noc_async_write_multicast_loopback_src(
                l1_addr, mcast_data_addr, nbytes, num_dests, false, noc);
        } else {
            noc_async_write_multicast(l1_addr, mcast_data_addr, nbytes,
                                      num_dests, false, noc);
        }

#ifdef ARCH_BLACKHOLE
        noc_async_writes_flushed(noc);
#endif

        // Multicast VALID signal (async - no barrier)
        *receiver_sem_ptr = VALID;
        uint64_t mcast_signal_addr =
            noc == 0
                ? get_noc_multicast_addr(x0_virtual.x, x0_virtual.y,
                                         x1_virtual.x, x1_virtual.y,
                                         receiver_sem_addr, noc)
                : get_noc_multicast_addr(x1_virtual.x, x1_virtual.y,
                                         x0_virtual.x, x0_virtual.y,
                                         receiver_sem_addr, noc);
        if (sender_in_range) {
            noc_semaphore_set_multicast_loopback_src(
                receiver_sem_addr, mcast_signal_addr, num_dests, false, noc);
        } else {
            noc_semaphore_set_multicast(receiver_sem_addr, mcast_signal_addr,
                                        num_dests, false, noc);
        }
        // NO barrier - return immediately
    } else if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
        // RECEIVER: Signal ready but don't wait for data
        noc_semaphore_set(receiver_sem_ptr, INVALID);
        uint64_t sender_sem_noc_addr = get_noc_addr(
            sender_virtual.x, sender_virtual.y, sender_sem_addr, noc);
        noc_semaphore_inc(sender_sem_noc_addr, 1, noc);
        // NO wait - return immediately
    }
}

// Wait for broadcast_async() to complete
// SENDER: Waits for write operations to complete
// RECEIVER: Waits for VALID signal from sender
FORCE_INLINE void broadcast_wait(uint32_t sender_x, uint32_t sender_y,
                                 uint32_t x0, uint32_t y0, uint32_t x1,
                                 uint32_t y1, uint32_t receiver_sem_id,
                                 uint8_t noc) {
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);
    volatile tt_l1_ptr uint32_t *receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(receiver_sem_addr);

    const uint32_t x = worker_logical_x();
    const uint32_t y = worker_logical_y();
    if (sender_x == x0 && sender_x == x1 && sender_y == y0 && sender_y == y1) {
        return;
    }

    if (x == sender_x && y == sender_y) {
        // SENDER: Wait for all writes to complete
        noc_async_write_barrier(noc);
    } else if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
        // RECEIVER: Wait for VALID signal
        noc_semaphore_wait(receiver_sem_ptr, VALID);
    }
}

FORCE_INLINE void sync_tensix_cores(uint32_t master_x, uint32_t master_y,
                                    uint32_t slave_x0, uint32_t slave_y0,
                                    uint32_t slave_x1, uint32_t slave_y1,
                                    uint32_t master_sem, uint32_t slave_sem,
                                    uint8_t noc) {
    // Translate semaphore IDs to addresses and pointers
    uint32_t master_sem_addr = get_semaphore(master_sem);
    uint32_t slave_sem_addr = get_semaphore(slave_sem);
    volatile tt_l1_ptr uint32_t *master_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(master_sem_addr);
    volatile tt_l1_ptr uint32_t *slave_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(slave_sem_addr);

    // Get current core's logical coordinates
    const uint32_t x = worker_logical_x();
    const uint32_t y = worker_logical_y();

    const auto master_virtual = worker_virtual_coord(master_x, master_y);
    const auto x0_virtual = worker_virtual_coord(slave_x0, slave_y0);
    const auto x1_virtual = worker_virtual_coord(slave_x1, slave_y1);

    // Check if master is in destination range
    bool master_in_range = (master_x >= slave_x0 && master_x <= slave_x1 &&
                            master_y >= slave_y0 && master_y <= slave_y1);

    uint32_t num_dests = (slave_x1 - slave_x0 + 1) * (slave_y1 - slave_y0 + 1);

    if (x == master_x && y == master_y) {
        // MASTER: Wait for all slaves to signal ready
        noc_semaphore_wait(master_sem_ptr,
                           master_in_range ? num_dests - 1 : num_dests);
        noc_semaphore_set(master_sem_ptr, 0);

        // Multicast VALID signal to all slaves (possibly including master)
        *slave_sem_ptr = VALID;
        uint64_t mcast_signal_addr =
            noc == 0
                ? get_noc_multicast_addr(x0_virtual.x, x0_virtual.y,
                                         x1_virtual.x, x1_virtual.y,
                                         slave_sem_addr, noc)
                : get_noc_multicast_addr(x1_virtual.x, x1_virtual.y,
                                         x0_virtual.x, x0_virtual.y,
                                         slave_sem_addr, noc);
        if (master_in_range) {
            noc_semaphore_set_multicast_loopback_src(
                slave_sem_addr, mcast_signal_addr, num_dests, false, noc);
        } else {
            noc_semaphore_set_multicast(slave_sem_addr, mcast_signal_addr,
                                        num_dests, false, noc);
        }
        noc_async_write_barrier(noc);
    } else if (x >= slave_x0 && x <= slave_x1 && y >= slave_y0 &&
               y <= slave_y1) {
        // SLAVE: This core is in destination range
        noc_semaphore_set(slave_sem_ptr, INVALID);

        // Signal master that we're ready
        uint64_t master_sem_noc_addr = get_noc_addr(
            master_virtual.x, master_virtual.y, master_sem_addr, noc);
        noc_semaphore_inc(master_sem_noc_addr, 1, noc);

        // Wait for master to signal data is ready
        noc_semaphore_wait(slave_sem_ptr, VALID);
    }
    // Else: Core is neither master nor slave - do nothing
}

// Intracore sync between BRISC and NCRISC using circular buffers
FORCE_INLINE void dmvk_barrier() {
    uint32_t sync_cb1_id = tt::CBIndex::c_25;
    uint32_t sync_cb2_id = tt::CBIndex::c_26;
#ifdef COMPILE_FOR_NCRISC
    cb_push_back(sync_cb1_id, 1);
    cb_wait_front(sync_cb2_id, 1);
    cb_pop_front(sync_cb2_id, 1);
#endif
#ifdef COMPILE_FOR_BRISC
    cb_push_back(sync_cb2_id, 1);
    cb_wait_front(sync_cb1_id, 1);
    cb_pop_front(sync_cb1_id, 1);
#endif
}

FORCE_INLINE void sync_all(uint32_t active_pw, uint32_t active_ph,
                           uint32_t global_master_sem,
                           uint32_t global_slave_sem,
                           uint8_t sync_noc = 0) {
    dmvk_barrier();
    if (noc_index == sync_noc) {
        sync_tensix_cores(0, 0, 0, 0, active_pw - 1, active_ph - 1,
                          global_master_sem, global_slave_sem, sync_noc);
    }
    dmvk_barrier();
}

FORCE_INLINE uint32_t min(uint32_t value1, uint32_t value2) {
    return value1 < value2 ? value1 : value2;
}
