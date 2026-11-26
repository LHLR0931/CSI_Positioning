using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace CsiApp
{
    /// <summary>Portrait virtual navigation. AR is a separate real-camera scene.</summary>
    public sealed class ApartmentMobilePresentationUi : MonoBehaviour
    {
        private ApartmentPresentationController controller;
        private GameObject canvasObject, drawerRoot, lookControls;
        private Camera background;
        private RectTransform safe;
        private ApartmentMobileLookPad lookPad;
        private Text nextText, summaryText, statusText, viewHint, playText, sourceText;
        private Font font;
        private Rect lastSafe;
        private Vector2 lastScreen;
        private bool drawerOpen;
        private string draftStartPlace = "grid", draftDestination = "balcony";
        private int draftStart, draftTarget = 14;
        private Text draftStartText, draftTargetText, draftNote;
        private GameObject startStepper, targetStepper;
        private readonly Dictionary<string, Image> startChoices = new Dictionary<string, Image>();
        private readonly Dictionary<string, Image> destinationChoices = new Dictionary<string, Image>();
        private readonly Dictionary<ApartmentPresentationController.ViewMode, Image> viewChoices = new Dictionary<ApartmentPresentationController.ViewMode, Image>();
        private static readonly Color Ink = new Color32(23, 55, 52, 255), Paper = new Color32(243, 246, 241, 255),
            Teal = new Color32(0, 120, 105, 255), Pale = new Color32(222, 235, 225, 255), Muted = new Color32(91, 113, 104, 255),
            Selected = new Color32(169, 217, 199, 255), Amber = new Color32(145, 91, 29, 255);

        public void Build(ApartmentPresentationController owner)
        {
            controller = owner; font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (background == null)
            {
                var go = new GameObject("Mobile model background", typeof(Camera)); go.transform.SetParent(transform, false);
                background = go.GetComponent<Camera>(); background.depth = -20; background.cullingMask = 0;
                background.clearFlags = CameraClearFlags.SolidColor; background.backgroundColor = Paper;
            }
            if (Application.isPlaying && GetComponentInChildren<EventSystem>() == null)
            {
                var go = new GameObject("Mobile presentation events", typeof(EventSystem), typeof(StandaloneInputModule));
                go.transform.SetParent(transform, false); go.GetComponent<EventSystem>().sendNavigationEvents = false;
            }
            Rebuild();
        }
        private void Update()
        {
            if (controller != null && (lastSafe != Screen.safeArea || lastScreen != new Vector2(Screen.width, Screen.height))) Rebuild();
        }
        private void Rebuild()
        {
            CancelGesture();
            if (canvasObject != null)
            {
                canvasObject.SetActive(false);
                if (Application.isPlaying) Destroy(canvasObject); else DestroyImmediate(canvasObject);
            }
            drawerRoot = null; viewChoices.Clear(); startChoices.Clear(); destinationChoices.Clear();
            lastSafe = Screen.safeArea; lastScreen = new Vector2(Screen.width, Screen.height);
            canvasObject = new GameObject("Portrait virtual navigation UI", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            canvasObject.layer = 5; canvasObject.transform.SetParent(transform, false);
            var canvas = canvasObject.GetComponent<Canvas>(); canvas.renderMode = RenderMode.ScreenSpaceCamera;
            canvas.worldCamera = controller.interfaceCamera; canvas.planeDistance = .6f;
            controller.interfaceCamera.rect = new Rect(0, 0, 1, 1); controller.interfaceCamera.ResetAspect();
            var scaler = canvasObject.GetComponent<CanvasScaler>(); scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(390, 844); scaler.matchWidthOrHeight = 0;
            canvas.scaleFactor = Mathf.Max(1, Screen.width) / 390f;
            safe = new GameObject("Phone safe area", typeof(RectTransform)).GetComponent<RectTransform>();
            safe.gameObject.layer = 5; safe.SetParent(canvasObject.transform, false);
            safe.anchorMin = new Vector2(lastSafe.xMin / Mathf.Max(1, Screen.width), lastSafe.yMin / Mathf.Max(1, Screen.height));
            safe.anchorMax = new Vector2(lastSafe.xMax / Mathf.Max(1, Screen.width), lastSafe.yMax / Mathf.Max(1, Screen.height));
            safe.offsetMin = safe.offsetMax = Vector2.zero; Canvas.ForceUpdateCanvases();
            float w = safe.rect.width, h = safe.rect.height, margin = 12, contentWidth = w - 24;
            const float headerHeight = 126, footerHeight = 206;
            float footerY = Mathf.Max(headerHeight + 110, h - footerHeight);
            Panel(safe, "Header", 0, 0, w, headerHeight, Paper);
            Button(safe, PlanningPreviewTransfer.ReturnToPlanning ? "Map" : "Home", 12, 6, 60, 38, controller.ReturnHome, Pale, Ink);
            Label(safe, "WAYFINDING", 84, 14, w - 198, 27, 18, Ink, true);
            if (PlanningPreviewTransfer.ReturnToPlanning) Button(safe, "Back to plan", w - 104, 6, 92, 38, controller.ReturnHome, Teal, Color.white);
            else Button(safe, "Home", w - 104, 6, 92, 38, controller.ReturnHome, Teal, Color.white);
            Label(safe, "Virtual view / simulated position", 14, 51, w - 28, 22, 12, Muted);
            float viewWidth = (contentWidth - 12) / 3;
            AddViewButton("First person", ApartmentPresentationController.ViewMode.FirstPerson, margin, 80, viewWidth);
            AddViewButton("Overview", ApartmentPresentationController.ViewMode.Overview, margin + viewWidth + 6, 80, viewWidth);
            AddViewButton("Top view", ApartmentPresentationController.ViewMode.Plan, margin + (viewWidth + 6) * 2, 80, viewWidth);
            float viewportHeight = Mathf.Max(90, footerY - headerHeight - 8);
            var viewport = Box(safe, "Touch model view", margin, headerHeight, contentWidth, viewportHeight);
            var touchImage = viewport.gameObject.AddComponent<Image>(); touchImage.color = Color.clear; touchImage.raycastTarget = true;
            lookPad = viewport.gameObject.AddComponent<ApartmentMobileLookPad>(); lookPad.Controller = controller;
            Panel(viewport, "View notice background", 0, 0, contentWidth, 61, new Color32(243, 246, 241, 245));
            viewHint = Label(viewport, "", 9, 7, contentWidth - 18, 22, 12, Ink);
            lookControls = Box(viewport, "Virtual look controls", 0, viewportHeight - 40, contentWidth, 36).gameObject;
            float lookWidth = (contentWidth - 12) / 3;
            Button(lookControls.transform, "Look left", 0, 0, lookWidth, 36, () => controller.RotateVirtualView(-25, 0), Pale, Ink);
            Button(lookControls.transform, "Face route", lookWidth + 6, 0, lookWidth, 36, controller.RecenterVirtualView, Pale, Ink);
            Button(lookControls.transform, "Look right", (lookWidth + 6) * 2, 0, lookWidth, 36, () => controller.RotateVirtualView(25, 0), Pale, Ink);
            var footer = Box(safe, "Navigation card", margin, footerY, contentWidth, footerHeight - 4);
            var footerImage = footer.gameObject.AddComponent<Image>(); footerImage.color = Color.white; footerImage.raycastTarget = true;
            nextText = Label(footer, "", 12, 7, contentWidth - 24, 43, 19, Ink, true);
            summaryText = Label(footer, "", 12, 50, contentWidth - 24, 36, 12, Muted);
            float buttonWidth = (contentWidth - 30) / 2;
            if (PlanningPreviewTransfer.ReturnToPlanning) Button(footer, "Back to plan", 12, 90, buttonWidth, 42, controller.ReturnHome, Pale, Ink);
            else Button(footer, "Route options", 12, 90, buttonWidth, 42, OpenDrawer, Pale, Ink);
            playText = Button(footer, "Play route", 18 + buttonWidth, 90, buttonWidth, 42, controller.TogglePlayback, Teal, Color.white);
            Button(footer, "Reset", 12, 139, buttonWidth, 35, controller.ResetRoute, Pale, Ink);
            Button(footer, "End demo", 18 + buttonWidth, 139, buttonWidth, 35, controller.EndRoute, Pale, Ink);
            sourceText = Label(footer, "", 12, 179, contentWidth - 24, 20, 10, Amber);
            statusText = Label(viewport, "", 9, 31, contentWidth - 18, 22, 11, Muted);
            Canvas.ForceUpdateCanvases();
            var corners = new Vector3[4]; viewport.GetWorldCorners(corners);
            Vector2 lower = RectTransformUtility.WorldToScreenPoint(controller.interfaceCamera, corners[0]);
            Vector2 upper = RectTransformUtility.WorldToScreenPoint(controller.interfaceCamera, corners[2]);
            controller.SetMobileViewport(new Rect(lower.x / Screen.width, lower.y / Screen.height,
                (upper.x - lower.x) / Screen.width, (upper.y - lower.y) / Screen.height));
            if (drawerOpen) DrawDrawer(); Refresh();
        }
        private void AddViewButton(string title, ApartmentPresentationController.ViewMode mode, float x, float y, float width)
        {
            var label = Button(safe, title, x, y, width, 36, () => controller.SetView(mode), Pale, Ink);
            viewChoices[mode] = label.transform.parent.GetComponent<Image>();
        }
        private void OpenDrawer()
        {
            if (drawerOpen) return;
            controller.PausePlayback("Choose a virtual start and destination.");
            draftStartPlace = controller.StartPlaceId; draftStart = controller.StartLabel;
            draftDestination = controller.DestinationId; draftTarget = controller.TargetLabel;
            drawerOpen = true; CancelGesture(); DrawDrawer();
        }
        public bool CloseDrawer()
        {
            if (!drawerOpen) return false;
            drawerOpen = false; CancelGesture();
            if (drawerRoot != null)
            {
                drawerRoot.SetActive(false);
                if (Application.isPlaying) Destroy(drawerRoot); else DestroyImmediate(drawerRoot);
            }
            drawerRoot = null; startChoices.Clear(); destinationChoices.Clear(); return true;
        }
        private void DrawDrawer()
        {
            float w = safe.rect.width, h = safe.rect.height;
            var overlay = Box(safe, "Route drawer", 0, 0, w, h); drawerRoot = overlay.gameObject;
            var dim = overlay.gameObject.AddComponent<Image>(); dim.color = new Color(0, .08f, .05f, .30f);
            var dismiss = overlay.gameObject.AddComponent<Button>(); dismiss.targetGraphic = dim; dismiss.onClick.AddListener(() => CloseDrawer());
            const float height = 370;
            var sheet = Box(overlay, "Route options sheet", 8, Mathf.Max(6, h - height - 4), w - 16, height);
            var sheetImage = sheet.gameObject.AddComponent<Image>(); sheetImage.color = Paper; sheetImage.raycastTarget = true;
            float width = w - 16;
            Label(sheet, "Route options", 16, 12, width - 76, 28, 21, Ink, true);
            Button(sheet, "Close", width - 68, 8, 56, 36, () => CloseDrawer(), Pale, Ink);
            Label(sheet, "Virtual start / not your current position", 16, 44, width - 32, 22, 12, Muted);
            ChoiceRow(sheet, 75, width, true);
            startStepper = Box(sheet, "Grid start stepper", 16, 123, width - 32, 36).gameObject;
            Button(startStepper.transform, "-", 0, 0, 40, 36, () => { draftStart = (draftStart + 14) % 15; RefreshDraft(); }, Pale, Ink);
            draftStartText = Label(startStepper.transform, "", 48, 5, width - 128, 28, 16, Ink, true); draftStartText.alignment = TextAnchor.MiddleCenter;
            Button(startStepper.transform, "+", width - 72, 0, 40, 36, () => { draftStart = (draftStart + 1) % 15; RefreshDraft(); }, Pale, Ink);
            Label(sheet, "DESTINATION", 16, 169, width - 32, 20, 12, Muted, true);
            ChoiceRow(sheet, 196, width, false);
            targetStepper = Box(sheet, "Grid target stepper", 16, 244, width - 32, 36).gameObject;
            Button(targetStepper.transform, "-", 0, 0, 40, 36, () => { draftTarget = (draftTarget + 14) % 15; RefreshDraft(); }, Pale, Ink);
            draftTargetText = Label(targetStepper.transform, "", 48, 5, width - 128, 28, 16, Ink, true); draftTargetText.alignment = TextAnchor.MiddleCenter;
            Button(targetStepper.transform, "+", width - 72, 0, 40, 36, () => { draftTarget = (draftTarget + 1) % 15; RefreshDraft(); }, Pale, Ink);
            draftNote = Label(sheet, "", 16, 284, width - 32, 30, 11, Amber);
            Button(sheet, "Apply route", 16, 323, width - 32, 39, ApplyDraft, Teal, Color.white);
            RefreshDraft();
        }
        private void ChoiceRow(Transform sheet, float y, float width, bool isStart)
        {
            string[] ids = { "grid" }, titles = { "Grid" };
            float bw = (width - 50) / 4;
            for (int i = 0; i < ids.Length; i++)
            {
                string id = ids[i];
                var label = Button(sheet, titles[i], 16 + i * (bw + 6), y, bw, 38,
                    () => { if (isStart) draftStartPlace = id; else draftDestination = id; RefreshDraft(); }, Pale, Ink);
                (isStart ? startChoices : destinationChoices)[id] = label.transform.parent.GetComponent<Image>();
            }
        }
        private void RefreshDraft()
        {
            foreach (var item in startChoices) item.Value.color = item.Key == draftStartPlace ? Selected : Pale;
            foreach (var item in destinationChoices) item.Value.color = item.Key == draftDestination ? Selected : Pale;
            startStepper.SetActive(draftStartPlace == "grid"); targetStepper.SetActive(draftDestination == "grid");
            draftStartText.text = "Grid " + draftStart + "  " + GridMap.Index(draftStart);
            draftTargetText.text = "Grid " + draftTarget + "  " + GridMap.Index(draftTarget);
            draftNote.text = draftStartPlace == "grid" && draftDestination == "grid"
                ? "Only four-neighbor grid edges. No live CSI."
                : "Demo entrances / site verification pending. Amber = illustrative segment.";
        }
        private void ApplyDraft()
        {
            if (draftStartPlace == "grid") controller.ConfigureDestination(draftStart, draftDestination, draftTarget);
            else controller.ConfigureFromPlace(draftStartPlace, draftDestination, draftTarget);
            CloseDrawer(); Refresh();
        }
        public void Refresh()
        {
            if (controller == null || nextText == null) return;
            var route = controller.ActiveRoute; nextText.text = controller.NextInstruction;
            summaryText.text = route == null ? "Choose a route" : route.StartName + " > " + route.DestinationName +
                "\n" + controller.RemainingDistance.ToString("F1") + " m remaining / " + route.Length.ToString("F1") + " m total";
            playText.text = controller.IsPlaying ? "Pause" : controller.HasEnded ? "Replay demo" : "Play route";
            statusText.text = controller.HasEnded ? "Demo ended" : controller.IsPlaying ? "Simulated walk in progress" : "Paused / drag to inspect";
            if (route != null && route.Points.Count < 2) statusText.text = "Start equals destination / choose another target";
            else if (controller.StatusMessage.Contains("not included")) statusText.text = "Requested scene is not included in this build";
            sourceText.text = route != null && route.UsesIllustrativeConnector
                ? "Demo entrance / site verification pending" : "Simulation / no live CSI / surveyed grid";
            bool first = controller.Mode == ApartmentPresentationController.ViewMode.FirstPerson;
            lookControls.SetActive(first);
            viewHint.text = first ? "Virtual camera / drag to look" : "Surveyed plan / cutaway view";
            foreach (var item in viewChoices) item.Value.color = item.Key == controller.Mode ? Selected : Pale;
        }
        public void CancelGesture() { if (lookPad != null) lookPad.Cancel(); }
        private void OnDisable() { CancelGesture(); }
        private static RectTransform Box(Transform parent, string name, float x, float y, float w, float h)
        {
            var go = new GameObject(name, typeof(RectTransform)); go.layer = 5; var r = go.GetComponent<RectTransform>(); r.SetParent(parent, false);
            r.anchorMin = r.anchorMax = r.pivot = new Vector2(0, 1); r.anchoredPosition = new Vector2(x, -y); r.sizeDelta = new Vector2(w, h); return r;
        }
        private static void Panel(Transform parent, string name, float x, float y, float w, float h, Color color)
        { var image = Box(parent, name, x, y, w, h).gameObject.AddComponent<Image>(); image.color = color; image.raycastTarget = false; }
        private Text Label(Transform parent, string value, float x, float y, float w, float h, int size, Color color, bool bold = false)
        {
            var text = Box(parent, "Text / " + value, x, y, w, h).gameObject.AddComponent<Text>(); text.font = font; text.text = value;
            text.fontSize = size; text.color = color; text.fontStyle = bold ? FontStyle.Bold : FontStyle.Normal;
            text.raycastTarget = false; text.alignment = TextAnchor.UpperLeft; text.horizontalOverflow = HorizontalWrapMode.Wrap; return text;
        }
        private Text Button(Transform parent, string title, float x, float y, float w, float h, UnityEngine.Events.UnityAction action, Color color, Color foreground)
        {
            var rect = Box(parent, "Button / " + title, x, y, w, h); var image = rect.gameObject.AddComponent<Image>(); image.color = color;
            var button = rect.gameObject.AddComponent<Button>(); button.targetGraphic = image; button.onClick.AddListener(action);
            var label = Label(rect, title, 0, 0, w, h, 13, foreground, true); label.alignment = TextAnchor.MiddleCenter; return label;
        }
    }
}
