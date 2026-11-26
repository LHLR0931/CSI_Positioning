# Apartment map module

This is an offline **virtual model preview**, not a camera feed, AR calibration, or live CSI integration.

## Geometry source

`Resources/Maps/apartment_v1_2026-10-10.json` is a byte-for-byte copy of the project survey source `maps/apartment_v1_2026-10-10.json`. Synchronize this copy deliberately when confirmed survey data changes; the runtime loader caches the resource for the current session.

- Plan `(x, y)` maps to Unity `(x, height, y)` in metres.
- Confirmed 15 points and 22 undirected four-neighbour edges are read from JSON.
- Regions use `min_xy_m`/`max_xy_m`; unsupported nested polygon arrays are deliberately omitted from the `JsonUtility` DTO.
- Unknown space has no generated floor. Office right edge is a dashed unspecified boundary.
- Real walls are zero-thickness planes at the confirmed net-space coordinates. The `.42 m` wall display height and `.26 m` furniture display height are **temporary cutaway styling**, not measurements. This module does not create collision volumes or full-height AR occluders.
- D01–D03 mark opening spans. They are not navigation destinations. No balcony exit, T01 position, or off-grid connecting route is invented.
- Source uncertainty and evidence remain in the original JSON. The runtime DTO intentionally exposes only fields needed to draw the preview.

## Integration

```csharp
var view = persistentRoot.AddComponent<CsiApp.ApartmentSceneView>();
view.Initialize(768, 768); // Synchronous; pass the intended image aspect ratio.
rawImage.texture = view.Texture;
view.SetTopView(true); // false = oblique 3D overview, not first person / AR.
view.SetRoute(pathLabels, predictedLabel, targetLabel);
```

`SetRoute` accepts `IList<int>`. Null/empty route draws no route; invalid/non-adjacent segments are skipped with a warning. A label of `-1` simply highlights no node for that role. Blue denotes the selected prediction and amber denotes the target; when they coincide the prediction colour takes precedence.

The preview uses a camera restricted to layer 30, a separate model root away from the main scene, and a `RenderTexture`. The camera is disabled for automatic rendering and renders only when initialized or updated. Keep the component on a persistent UI root rather than on a page rebuilt every click. Calling `Initialize` with a changed size replaces the texture, so reassign `rawImage.texture` afterwards. `Dispose()` releases owned graphics resources; `OnDestroy()` calls it automatically.

Other APIs:

- `ApartmentMap.Current`: loaded `SiteMapData`.
- `ApartmentMap.ToWorld(float x, float y, float height = 0)`.
- `SiteMapData.FindPoint(int label)`, `FindRegion(string id)`, `FindOpening(string id)`.
- `SiteMapData.HasEdge(int from, int to)` and `PlanBounds` (`Rect` in plan metres).

The shader is stored in Resources to retain it in builds. No physics, network, or AR provider is required by this module. Actual Editor and device rendering must be checked separately from C# compilation.
