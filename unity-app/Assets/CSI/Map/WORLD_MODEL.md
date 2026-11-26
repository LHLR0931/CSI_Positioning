# World-space apartment model

`ApartmentWorldModel` reads the same `ApartmentMap.Current` data as the offline preview. Its children use **1 local unit = 1 metre**, with plan `(x,y)` mapped to Unity `(x,height,y)`. It creates no camera, remote preview root, AR origin, or tracking logic. The owning controller supplies position, rotation, scale and camera.

```csharp
var owner = new GameObject("Apartment geometry");
owner.transform.SetParent(anchorTransform, false);
var model = owner.AddComponent<CsiApp.ApartmentWorldModel>();
model.Initialize(showFloors: true);
model.SetRoute(CsiApp.GridMap.Route(12, 0), 12, 0);
model.SetLabelsFacing(arCamera); // Repeat when the camera moves.
```

Other calls: `SetRenderLayer(int)` (default 0) and `Dispose()`; component destruction releases runtime-owned meshes, materials and its generated child root. Repeated `Initialize` toggles floors without duplicating geometry. `SetRoute` only draws configured four-neighbour edges and skips malformed segments. `Initialize(false)` hides opaque context floors for a future on-site overlay while retaining grid/boundary/route graphics.

The Inspector's `wallDisplayHeight = 0.42` and `furnitureDisplayHeight = 0.26` are **temporary cutaway presentation heights**, not measurements. Edit them before initialization, or regenerate the inspection model. No full-height occlusion/collision accuracy is claimed. Unknown space has no floor; unspecified office-right edge is dashed; opening centres are not navigation endpoints. T01 and the balcony exit remain unconfigured.

## Lab scale versus on-site scale

- A lab reference-image model may use owner `localScale = Vector3.one * 0.05f` (1:20). This deliberately shrinks the full model and does not establish field calibration.
- At local position zero, grid point 0 is at the anchor origin. To centre a miniature on its anchor, derive the offset from `ApartmentMap.Current.PlanBounds.center`; with an identity owner rotation, set local position to `-ApartmentMap.ToWorld(center.x, center.y) * scale`. Do not copy surveyed coordinates into the AR controller.
- On-site geometry may remain scale 1 only after the known physical marker pose is used to align map and AR coordinates. Merely detecting an image does not supply the missing site pose.

## Permanent inspection scene

Run **CSI → Generate field inspection model**, or batch method `CsiApp.Editor.ApartmentInspectionBuilder.BuildAll`. It creates `Assets/CSI/Scenes/ApartmentInspection.unity`, persistent meshes/materials under `Assets/CSI/Generated/ApartmentInspection/run_<UTC timestamp>/`, an overview camera, and `inspection-evidence.json` beside those assets. Open the resulting scene alone for review; generation uses an additive scene to avoid discarding unrelated edits.

The report records both source hashes, point/edge/opening counts, floor/asset counts and checks. Generation refuses mismatched source/resource data, stale cached DTO data, invalid labels, non-neighbour edges or solid boundaries overlapping an opening. Generated heights and geometry checks do not certify AR alignment, survey tolerance, CSI accuracy or physical navigation.

The saved scene retains the model component to restore built-in TextMesh font materials when reopened. `SealAsInspectionAssets()` is an Editor-builder lifecycle method, not an AR runtime API: persistent assets belong to the project and are not destroyed by the component.
