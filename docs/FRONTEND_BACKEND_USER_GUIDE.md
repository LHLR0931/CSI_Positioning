# Unity client user guide

The client separates live positioning, saved-result replay and AR alignment. Keep the displayed source visible when recording or presenting results.

## Connect to a live backend

1. Start the backend on the experiment computer and confirm that port `8765` is reachable.
2. Connect the phone and computer to the same network.
3. Open **Home → Settings → Connection settings**.
4. Enter `http://<computer-ip>:8765`, then choose **Save & connect**.
5. Check the backend state. Begin a live positioning run only when the expected model and session are shown and the state is `ready`.

Do not enter `0.0.0.0` on the phone. `localhost` refers to the phone unless an explicit USB port forwarding setup is active.

## Developer mode

Use **Developer → Replay** to inspect the bundled D1 predictions without a backend. Replay preserves the original windows, labels and model results; its playback speed is only a display setting.

Use **Developer → Live** to inspect predictions received from the configured service. A model button can only show a model actually returned by the data source. Do not assign an unnamed server prediction to a different model merely to fill the interface.

Ground truth must come from an independently known static label or from the saved replay record. Clear a static ground-truth label before the participant moves. The client must leave error unavailable when no matching truth exists.

## Plan and run a grid route

1. Open **Positioning**.
2. Select different start and target labels from 0–14.
3. Review the candidate route and choose one when alternatives are available.
4. Confirm the route. This fixes the plan but does not prove the participant is at the chosen start.
5. Move to the selected start and begin the start check.
6. Start the route only after fresh live windows agree with the start and the backend remains `ready`.
7. Follow the current-segment direction and next-grid indication. Pause if data becomes stale, the backend is degraded, or the session/model changes.
8. Confirm arrival only after independently checking that the participant has reached the target.

The route display must not advance from a repeated or cached prediction. After a pause, select the participant's actual route point and complete a fresh-data check before resuming.

## AR alignment

The 2D grid can be used without AR. For the AR view:

1. Fix the full reference image vertically with its top edge in the documented orientation.
2. Measure the image width, map-centre `x/y`, centre height and front-facing angle.
3. Save those measurements in **Image measurements**.
4. Choose **Scan image**, point the camera at the complete image, and wait for tracking.
5. Choose **Lock alignment**.
6. Compare the rendered positions of real floor points 0, 4 and 10 from more than one viewpoint.
7. Confirm each point only when its location, direction and scale agree with the physical grid.

If the reference image moves, tracking is lost, the app returns from the background, or the measurements change, repeat the alignment. AR is an overlay based on the measured reference pose; it is not independent ground truth for CSI accuracy.

Reference files and display instructions are in [`ar-reference/README.md`](ar-reference/README.md).

## Recording

Start recording only after verifying the source, session, selected model and any independent truth label. Stop and save before changing source, truth, connection or participant position.

The saved prediction rows describe what the client received. Navigation-motion rows may include held or AR-assisted positions and must not be treated as an independent trajectory truth. Preserve separate files for live, replay and controlled static experiments.

## Troubleshooting

| Symptom | Check |
|---|---|
| Cannot connect | Computer IP, port 8765, firewall, shared network and server bind address |
| Status is visible but no prediction | Calibration/training state, four-link health and HTTP 204 response |
| Prediction stops changing | `session_id`, `seq`, link state and packet capture at the Pi sources |
| A model remains unavailable | Whether that model is actually present in the response or replay file |
| Positioning pauses | Stale data, degraded state, changed session/model or lost AR tracking |
| Error is unavailable | No independent truth is associated with the current prediction |
| AR overlay is offset | Image width/pose, coordinate axes, floor height, facing angle and tracking stability |

Transport checks, AR alignment and accuracy evaluation are separate validations; success in one does not establish the others.
