import time
import random

# Adjust these imports if your folder structure requires it
import draft as draft
from env import dtp_platform

# =============================================================================
# MOCK ENVIRONMENT & CONFIGURATION
# =============================================================================
class MockEnv:
    """Mocks simpy.Environment to isolate class performance from sim loop overhead."""
    def __init__(self):
        self.now = 0

DUMMY_CONFIG = {
    "dtp_rules": {
        "slot_duration": 45,
        "priority_window": 10,
        "freeze_time": 60,
        "lead_time": 1440
    },
    "ghas": {
        "GHA_ALPHA": {"import": 15, "export": 15},
        "GHA_BETA":  {"import": 10, "export": 10}
    }
}

# =============================================================================
# DYNAMIC BENCHMARK ENGINE
# =============================================================================
def execute_benchmark(module, module_name: str, num_ops: int) -> float:
    """
    Runs a deterministic sequence of random operations against a specific module.
    Resetting the random seed at the start guarantees identical conditions.
    """
    random.seed(42)  # Ensures both engines get the exact same sequence of choices
    env = MockEnv()
    platform = module.DTPPlatform(env=env, cfg=DUMMY_CONFIG)

    trucks = [f"TRK_{i:05d}" for i in range(5000)]
    ghas = list(DUMMY_CONFIG["ghas"].keys())
    flows = ["import", "export"]
    
    # Track published slots locally to avoid KeyErrors on uninitialized keys
    published_slots = {gha: [] for gha in ghas}

    # Operation mix weights
    ops = [
        'publish_slot', 'book_slot', 'get_available_slots', 
        'get_booking', 'cancel_book', 'mark_docked', 
        'mark_closed', 'advance_time'
    ]
    weights = [20, 25, 20, 15, 10, 5, 4, 1]

    start_time = time.perf_counter()

    for _ in range(num_ops):
        op = random.choices(ops, weights=weights, k=1)[0]
        gha = random.choice(ghas)
        flow = random.choice(flows)
        truck = random.choice(trucks)

        # Calculate a window that respects the freeze_time and lead_time rules
        min_slot = ((env.now + DUMMY_CONFIG["dtp_rules"]["freeze_time"]) // 15 + 1) * 15
        max_slot = ((env.now + DUMMY_CONFIG["dtp_rules"]["lead_time"]) // 15 - 1) * 15
        slot = random.randrange(min_slot, max_slot + 1, 15) if min_slot < max_slot else min_slot

        if op == 'publish_slot':
            success = platform.publish_slot(gha, slot, flow)
            if success and slot not in published_slots[gha]:
                published_slots[gha].append(slot)

        elif op == 'book_slot':
            if published_slots[gha]:
                slot = random.choice(published_slots[gha])
            platform.book_slot(gha, slot, truck, flow)

        elif op == 'get_available_slots':
            platform.get_available_slots(gha, flow, 480)

        elif op == 'get_booking':
            platform.get_booking(gha, truck)

        elif op == 'cancel_book':
            if published_slots[gha]:
                slot = random.choice(published_slots[gha])
            platform.cancel_book(gha, slot, truck)

        elif op == 'mark_docked':
            if published_slots[gha]:
                slot = random.choice(published_slots[gha])
            platform.mark_docked(gha, slot, truck)

        elif op == 'mark_closed':
            if published_slots[gha]:
                slot = random.choice(published_slots[gha])
            platform.mark_closed(gha, slot, truck)

        elif op == 'advance_time':
            env.now += random.randint(1, 15)
            # Filter out expired slots from our selector pool
            for g in ghas:
                published_slots[g] = [s for s in published_slots[g] if s > env.now]

    end_time = time.perf_counter()
    return end_time - start_time

# =============================================================================
# RUNNER
# =============================================================================
if __name__ == "__main__":
    NUM_OPERATIONS = 1_000_000 
    print(f"Starting state-aware benchmark ({NUM_OPERATIONS:,} actions per module)...")
    
    time_dtp = execute_benchmark(dtp_platform, "dtp_platform.py (List/Scan version)", NUM_OPERATIONS)
    print(f" -> dtp_platform.py execution time: {time_dtp:.4f} seconds")
    
    time_draft = execute_benchmark(draft, "draft.py (Dict/Index version)", NUM_OPERATIONS)
    print(f" -> draft.py execution time: {time_draft:.4f} seconds")
    
    print("\n--- Results ---")
    if time_draft < time_dtp:
        improvement = ((time_dtp - time_draft) / time_dtp) * 100
        speedup = time_dtp / time_draft
        print(f"WINNER: draft.py is FASTER by {improvement:.2f}%")
        print(f"It processed the workload {speedup:.2f}x faster than dtp_platform.py.")
    else:
        improvement = ((time_draft - time_dtp) / time_draft) * 100
        speedup = time_draft / time_dtp
        print(f"WINNER: dtp_platform.py is FASTER by {improvement:.2f}%")
        print(f"It processed the workload {speedup:.2f}x faster than draft.py.")