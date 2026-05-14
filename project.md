# Multi Agent Reinforcement Learning in Landside Air Cargo Truck Slot Optimization

## Module Dependencies

The modules of this project follow these dependencies:

```plaintext
PARENT             CHILD
----------------------------------------------------------------------------------------------------------------------
params         --> dtp_platform | infrastructure | demand | objects | road | service_time | kpi_tracker | schiphol_env
dtp_platform   --> objects | demand | kpi_tracker | schiphol_env
infrastructure --> objects | demand | kpi_tracker | schiphol_env
service_time   --> objects | demand | schiphol_env
road           --> demand | schiphol_env
objects        --> demand | schiphol_env
demand         --> schiphol_env
kpi_tracker    --> schiphol_env
```

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
| _validate_gha         | ─                | gha                                              | None       |
| _assign_slot          | ─                | gha, slot_start, truck_id                        | None       |
| _free_slot            | ─                | gha, slot_start, truck_id                        | bool       |
| _set_phase            | ─                | gha, slot_start, truck_id, phase                 | None       | 
| _is_docked            | ─                | gha, slot_start, truck_id                        | bool       |
| _taken_docks_at       | ─                | gha, new_start                                   | int        |
| _total_published_at   | ─                | gha, new_start                                   | int        |

### Module: infrastructure

*CheckpointID* None

*SensorEvent* None

*InfrastructureLayer*
| Method             | Called by    | Args                             | Returns           |
|--------------------|--------------|----------------------------------|-------------------|
| _log               | ─            | event                            | None              |
| gate_in            | demand       | sim_time, truck                  | None              |
| gate_out           | demand       | sim_time, truck                  | None              |
| tp3_in             | objects      | sim_time, truck                  | None              |
| tp3_out            | objects      | sim_time, truck                  | None              |
| gha_in             | objects      | sim_time, truck, gha_id          | None              |
| dock_start         | objects      | sim_time, truck, gha_id, dock_id | None              |
| dock_end           | objects      | sim_time, truck, gha_id, dock_id | None              |
| flush_step_buffer  | schiphol_env | —                                | List[SensorEvent] |
| get_all_events     | kpi_tracker  | —                                | List[SensorEvent] |

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
| _dock_pool              | ─         | flow_type                 | simpy.Resource |
| _queue                  | ─         | flow_type                 | simpy.Resource |
| process_truck           | demand    | truck, dtp                | None           |
| release_window_watcher  | ─         | slot_start, dtp, tp3      | None           |
| exp_occupancy           | ─         | ─                         | float          |
| imp_occupancy           | ─         | ─                         | float          |
| exp_queue_norm          | ─         | ─                         | float          |
| imp_queue_norm          | ─         | ─                         | float          |
| upcoming_bookings_norm  | ─         | dtp, horizon              | float          |

*TP3Buffer*
| Method                     | Called by    | Args                         | Returns       |
|----------------------------|--------------|------------------------------|---------------|
| enter                      | demand       | truck                        | None          |
| release                    | demand       | truck_id                     | Truck|None    |
| release_next               | ─            | gha                          | Truck|None    |
| signal_standby_opportunity | ─            | gha, slot_start, signal_time | List[Dict]    |
| get_pending_signals        | demand       | —                            | List[Dict]    |
| occupancy_ratio            | ─            | —                            | float         |
| n_parked                   | ─            | ─                            | int           |
| n_overflow                 | ─            | —                            | int           |
| parked_by_flow_type        | ─            | flow_type                    | int           |
| get_parked_trucks          | ─            | —                            | List[Truck]   |

### Module: road

| Method           | Called by | Args       | Returns |
|------------------|-----------|------------|---------|
| _apply_noise     | ─         | base_time  | float   |
| time_from_to     | demand    | start, end | float   |

### Module: service_time

| Method    | Called by | Args      | Returns    |
|-----------|-----------|-----------|------------|
| sample    | objects   | flow_type | float      |
| mean      | ─         | flow_type | float      |

### Module: demand

| Method                | Called by | Args        | Returns |
| ----------------------|-----------|-------------|---------|
| run                   | ─         | ─           | None    |
| _rate_at              | ─         | t           | float   |
| _create_truck         | ─         | ─           | Truck   |
| _book_slots           | ─         | truck       | bool    |
| _cancel_all           | ─         | truck       | None    |
| _truck_journey        | ─         | truck       | None    |
| _handle_tp3_redirect  | ─         | truck, gha  | None    |
| _origin_to_gate       | ─         | origin_type | float   |
| _intra_airport_buffer | ─         | ─           | float   |

---

## Glossary

### Slot

A `slot` is just a dictionary with 2 keys: `truck_id` and `phase`. Multiple slots can fit into a list of dictionaries if they all start at the same time. When it is firstly published it does not contain the truck id. The slot is a different concept from the booking.
```python
slot = {"truck_id": None, "phase": "available"}
```
