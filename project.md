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

## Hyperparameter tuning

```Plaintext

1440 steps x 6 agents = 8640 frames_per_episode

# We need to aim at 16-24 episodes_per_batch
frames_per_batch = 16 episodes_per_batch x 8640 frames_per_episode = 138_240 --> We round up to 144_000

# In the last run one sequential worker collected 5333 steps in 173s
5333 / 173 ~ 30 steps_per_second

# One full 1440 steps episode should take up:
1440 / 30 ~ 48s

# The lxc container in which this runs has 7 cpu threads, so leaving 1 for overhead we are left with 6.
6 workers x 1 env each = 6 parallel episodes

# Since one batch has 16 episodes, the time to complete one batch with 6 workers is:
ceil(16/6) x 45s ~ 2 x 45 = 90s
```

The experiment settings has to be:

```python

on_policy_minibatch_size=8640    # one full episode
on_policy_n_minibatch_iters=16    # we want 16 episodes per batch
```

---

## Action space

### Transporter

| Range | Action                                               | Count |
| ---   | ---                                                  | ---   |
| 0     | no_op                                                | 1     |
| 1-40  | Book truck `(idx-1)//N_GHAS` at GHA `(idx-1)%N_GHAS` | 40    |
| 41-50 | Dispatch truck `action-41`                           | 10    |
| Total |                                                      | 51    |

### Ghas

Calculations:
- Slot duration = 45 min
- Publishing horizon = 72 h = 4320 min
- Slots = 4320 / 45 = 96 slots

| Range  | Action                                  | Count |
| ---    | ---                                     | ---   |
| 0      | no_op                                   | 1     |
| 1-96   | Publish `export` slots                  | 96    |
| 97-192 | Publish `import` slots                  | 96    |
| Total  |                                         | 193   |

### Orchestrator

| Offset                | Value |
| ---                   | ---   |
| _ORCH_BOOK_OFFSET     | 1     |
| _ORCH_DISPATCH_OFFSET | 41    |
| _ORCH_CANCEL_OFFSET   | 51    |
| _ORCH_MODIFY_OFFSET   | 91    |
| last action           | 250   |

| Range  | Action                                                          | Count |
| ---    | ---                                                             | ---   |
| 0      | no_op                                                           | 1     |
| 1-40   | Book truck `t` at gha `g`                                       | 40    |
| 41-50  | Dispatch truck `t` from origin                                  | 10    |
| 51-90  | Cancel booking of truck `t` at gha `g`                          | 40    |
| 91-250 | Modify truck `t` booking `from_gha, to_gha, from_slot, to_slot` | 160   |
| Total  |                                                                 | 251   |