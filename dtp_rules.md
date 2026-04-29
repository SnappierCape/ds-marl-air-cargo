# DTP Rules:

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
11. GHAs can remove from the DTP platform unbooked published slot outside of the frozen window.
12. For each GHA, the total amount of docks is split equally among export and import. GHAs can not have an odd number of docks.