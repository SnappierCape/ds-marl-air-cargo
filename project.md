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

## Glossary

### Slot

A `slot` is just a dictionary with 2 keys: `truck_id` and `phase`. Multiple slots can fit into a list of dictionaries if they all start at the same time. When it is firstly published it does not contain the truck id. The slot is a different concept from the booking.
```python
slot = {"truck_id": None, "phase": "available"}
```
