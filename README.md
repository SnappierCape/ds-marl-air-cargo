# Multi Agent Reinforcement Learning in Landside Air Cargo Truck Slot Optimization

## Module Dependencies

The modules of this project follow these dependencies:

```plaintext
PARENT             CHILD
-----------------------------------------------------------------------------------------
params         --> dtp_platform | infrastructure | demand | objects | road | service_time
dtp_platform   --> objects | demand
infrastructure --> objects | demand
service_time   --> objects | demand
road           --> demand
objects        --> demand
demand         --> None
```

---

## Module Descriptions

- **params:** The central parameters configuration file, it holds all the hardcoded values that the project needs. All the other modules draw values from this module to make sure there are no hardcoded values in the scripts.
- **dtp_platform:** Handles the slot booking logic, tracks the state of the slots and the bookings, and enforces the slot booking DTP rules.
- **infrastructure:** Ckeckpoint tracking.
- **service_time:** Samples a single dock service time from a statistical distribution, ensuring volatility in the time that the trucks spend at the gha.
- **road:** Generates travel times between each node of the road network based on real life distances and applies a stocastic noise to those values.
- **objects:** Defines the 3 most critical SimPy entities: `Truck`, `TP3Buffer` and `GHATerminal`.
- **demand:** Creates the trucks, books the slots, and sends the truck at the Airport following the average travel times for each source. The truck creation rate depends on the time of the day.

---

## DTP Rules:

The project is based on the following rules / assumptions:
  1. Every truck must have a confirmed DTP booking to pass the Main Gate. Enforced by ANPR at the Main Gate. No exceptions.
  2. Slots are 45-minutes time windows. One slot = one license plate = one time frame = one ground handler. One truck per slot.
  3. All GHAs publish their slots on the shared DTP platform. Visible to all participants in real time. This is the only way to publish a slot.
  4. DTP platform is operated by a neutral third party (Schiphol/ACN).
  5. The Orchestrator has full authority to cancel or modify any booking unilaterally as long as the truck isn't already docked, without requiring Transporter or GHA acceptance. It can NOT remove a published slot from the DTP platform.
  6. Minimum booking lead time is double the slot duration. After that slots are frozen: no new bookings, no cancellations. this is called the "frozen window".
  7. GHAs may publish slots up to 72h before the start of the slot and before the frozen window starts.
  8. Slots are divided into a "priority window" (first 10m) and a "release window" (m11 to slot end), the dock is held still until minute 10. Then the slot is available for the next trucks at the GHA queue or the trucks sitting at TP3. The DTP platform or the Orchestrator may release them. If the original truck shows up between minute 11 and slot end:
     - If the dock is still free → the original truck is admitted directly. No rebooking required. A small late-penalty is logged against the transporter account for RL feedback purposes but the truck proceeds.
     - If the dock is not free → the original truck is redirected to TP3 as a standby truck. It does not need to rebook a new slot. Its existing booking remains valid and it re-enters the queue when the Orchestrator or its own timing releases it.
  9. A truck that shows up after its own slot has expired is recorded as "no show" and the penalty is logged for RL feedback purposes. The truck is redirected at tp3. It can book another slot or the Orchestrator can send it to a GHA.
  10. The transporter can cancel a booking outside of the frozen window, only the Orchestrator can cancel a booking inside the froozen window.
  11. GHAs can not remove a published slot.
  12. For each GHA, the total amount of docks is split equally among export and import. GHAs can not have an odd number of docks.

---

## Module Interfaces

### Module: dtp_platform

| Method                | Called by        | Args                                             | Returns    |
|-----------------------|------------------|--------------------------------------------------|------------|
| publish_slot          | ─                | gha, slot_start                                  | bool       |
| book_slot             | demand           | gha, slot_start, truck_id                        | bool       |
| orch_book_slot        | ─                | gha, slot_start, truck_id                        | bool       |
| cancel_book           | demand           | gha, slot_start, truck_id                        | bool       |
| orch_cancel_book      | ─                | gha, slot_start, truck_id                        | bool       |
| orch_modify_book      | ─                | truck_id, from_gha, from_start, to_gha, to_start | bool       |
| get_slot_phase        | objects          | slot_start, arrival_time, dock_is_free           | str        |
| release_to_standby    | objects          | gha, slot_start                                  | bool       |
| mark_docked           | objects          | gha, slot_start, truck_id                        | None       |
| mark_closed           | objects          | gha, slot_start, truck_id                        | None       |
| record_late           | objects          | truck_id                                         | None       |
| record_no_show        | objects          | gha, slot_start, truck_id                        | None       |
| get_available_slots   | demand           | gha, horizon                                     | List[int]  |
| get_booking           | demand           | gha, truck_id                                    | int|None   |
| count_available_slots | ─                | gha, horizon                                     | int        |
| _free_slot            | ─                | gha, slot_start, truck_id                        | bool       | 
| _is_docked            | ─                | gha, slot_start, truck_id                        | bool       | 

### Module: infrastructure

| Method             | Called by  | Args                        | Returns          |
|--------------------|------------|-----------------------------|------------------|
| gate_in            | demand     | sim_time, truck             | None             |
| gate_out           | demand     | sim_time, truck             | None             |
| tp3_in             | objects    | sim_time, truck             | None             |
| tp3_out            | objects    | sim_time, truck             | None             |
| gha_in             | objects    | sim_time, truck, gha        | None             |
| dock_start         | objects    | sim_time, truck, gha, dock_id | None           |
| dock_end           | objects    | sim_time, truck, gha, dock_id | None           |
| flush_step_buffer  | env        | —                           | List[SensorEvent]|
| get_all_events     | kpi_tracker| —                          | List[SensorEvent]|

### Module: objects

*Truck*
| Method          | Called by      | Args | Returns   |
|-----------------|----------------|------|-----------|
| total_parcels   | infrastructure | ─    | int       |
| parcels_for     | infrastructure | gha  | int       |
| next_slot       | ─              | ─    | int|None  |
| next_stop       | ─              | ─    | Dict|None |
| complete_stop   | demand         | gha  | ─         |

*GHATerminal*
| Method                  | Called by | Args                      | Returns        |
|-------------------------|-----------|---------------------------|----------------|
| _route_dock_pool        | ─         | flow_type                 | simpy.Resource |
| _route_queue            | ─         | flow_type                 | simpy.Resource |
| process_truck           | demand    | truck, dtp                | None           |
| release_window_watcher  | ─         | slot_start, dtp, tp3      | None           |
| exp_occupancy           | ─         | ─                         | float          |
| imp_occupancy           | ─         | ─                         | float          |
| exp_queue_norm          | ─         | max_q                     | float          |
| imp_queue_norm          | ─         | max_q                     | float          |
| upcoming_bookings_norm  | ─         | dtp, horizon, max_b       | float          |

*TP3Buffer*
| Method                     | Called by    | Args                         | Returns       |
|----------------------------|--------------|------------------------------|---------------|
| enter                      | demand       | truck                        | None          |
| release                    | demand       | truck_id                     | Truck|None    |
| release_next               | ─            | gha                          | Truck|None    |
| signal_standby_opportunity | ─            | gha, slot_start, signal_time | None          |
| get_pending_signals        | demand       | —                            | List[Dict]    |
| occupancy_ratio            | ─            | —                            | float         |
| n_parked                   | ─            | ─                            | int           |
| n_overflow                 | ─            | —                            | int           |
| parked_by_flow_type        | ─            | flow_type                    | int           |
| get_parked_trucks          | ─            | —                            | List[Truck]   |

### Module: road

| Method           | Called by | Args      | Returns |
|------------------|-----------|-----------|---------|
| _apply_noise     | ─         | base_time | float   |
| gate_to_gha      | demand    | gha       | float   |
| gate_to_tp3      | demand    | —         | float   |
| tp3_to_gha       | demand    | gha       | float   |

### Module: service_time

| Method    | Called by | Args      | Returns    |
|-----------|-----------|-----------|------------|
| _validate | ─         | cfg       | None       |
| sample    | objects   | flow_type | float      |
| mean      | ─         | flow_type | float|None |