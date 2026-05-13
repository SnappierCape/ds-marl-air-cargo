# scripts/register_task.py
from benchmarl.environments import TaskRegistry
from marl.benchmarl_task import SchipholTask

# Register both scenarios under the "schiphol" environment namespace
TaskRegistry.register_task("schiphol", SchipholTask)