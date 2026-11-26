using System;
using System.Collections;
using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.SpatialTracking;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.XR.Management;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

namespace CsiApp
{
    /// <summary>
    /// Real camera tracking and surveyed, full-scale image-to-map registration.
    /// This component creates no navigation UI, simulated motion, or assumed site pose.
    /// MarkerMapPose must describe an upright, fixed wall image, not a horizontal image.
    /// </summary>
    public sealed class SiteArTracker : MonoBehaviour
    {
        // Installed AR Foundation 6.5.1, Documentation~/xr-simulation/
        // simulation-environments.md: image local +Y is the front surface normal,
        // local +Z points to the top of the image; its image plane is local XZ.
        // TrackedImageDiscoveryStrategy.cs also uses image.transform.up as normal.
        private const string ReferenceResource = "AR/CSI_Reference";
        private const string ReferenceName = "CSI Site Reference";
        private const float MountToleranceDegrees = 20f;
        private static readonly Vector3 InvalidVector = new Vector3(float.NaN, float.NaN, float.NaN);

        public bool Ready { get { return requested && ready && !booting && runtimeRoot != null; } }
        public bool IsBusy { get { return booting || imageJobRunning || anchorPending; } }
        public bool HasCandidate
        {
            get
            {
                string reason;
                return Ready && !paused && !needsRescan && !calibrated && !IsBusy
                    && DeviceTracking && candidate != null && ValidImage(candidate, out reason);
            }
        }
        public bool IsCalibrated
        { get { return calibrated && !needsRescan && mapRoot != null && mapRoot.gameObject.activeInHierarchy && UnitScale(mapRoot) && IsTracking; } }
        public bool IsVerified { get { return verified && IsCalibrated; } }
        public bool IsTracking
        {
            get
            {
                return DeviceTracking && (!calibrated ||
                    (anchor != null && anchor.trackingState == TrackingState.Tracking && UnitScale(anchor.transform)));
            }
        }
        public string Message { get; private set; } = "Camera AR is stopped. Real CSI can still use the 2D view.";
        public Transform MapRoot { get { return mapRoot; } }
        public UnityEngine.Camera Camera { get { return arCamera; } }
        /// <summary>Map metres (x, camera height, map y). Invalid without a live alignment.</summary>
        public Vector3 MapCameraPosition
        {
            get { return IsCalibrated ? mapRoot.InverseTransformPoint(arCamera.transform.position) : InvalidVector; }
        }
        /// <summary>Map-local horizontal heading. Looking exactly up/down gives no invented heading.</summary>
        public Vector3 MapCameraForward
        {
            get
            {
                if (!IsCalibrated) return InvalidVector;
                Vector3 forward = mapRoot.InverseTransformDirection(arCamera.transform.forward);
                forward.y = 0;
                return Finite(forward) && forward.sqrMagnitude > .0001f ? forward.normalized : InvalidVector;
            }
        }
        public event Action Changed;

        private XRManagerSettings xr;
        private GameObject runtimeRoot;
        private ARSession session;
        private ARCameraManager cameraManager;
        private ARTrackedImageManager images;
        private ARAnchorManager anchors;
        private ARAnchor anchor;
        private ARTrackedImage candidate;
        private UnityEngine.Camera arCamera;
        private Transform mapRoot;
        private readonly List<UnityEngine.Camera> suspendedCameras = new List<UnityEngine.Camera>();
        private Coroutine startupRoutine;
        private MutableRuntimeReferenceImageLibrary library;
        private Texture2D referenceTexture;
        private AddReferenceImageJobState imageJob;
        private Guid imageGuid;
        private SiteNavigationConfig sourceConfig;
        private Pose markerMapPose;
        private float imageWidth;
        private float measuredX, measuredY, measuredHeight, measuredFacing;
        private int generation, imageJobGeneration, anchorRequestId;
        private bool requested, ready, booting, paused, ownsLoader;
        private bool imageJobRunning, anchorPending, calibrated, verified, needsRescan = true;
        private bool hasCameraFrame, lastTracking;

        private bool CanPrepareImageLibrary
        {
            get
            {
                return !paused && hasCameraFrame && session != null && session.subsystem != null
                    && session.subsystem.running && (ARSession.state == ARSessionState.SessionInitializing
                        || ARSession.state == ARSessionState.SessionTracking);
            }
        }

        private bool DeviceTracking
        {
            get
            {
                return Ready && !paused && hasCameraFrame && ARSession.state == ARSessionState.SessionTracking
                    && arCamera != null && arCamera.isActiveAndEnabled && cameraManager != null
                    && cameraManager.enabled && cameraManager.permissionGranted
                    && cameraManager.subsystem != null && cameraManager.subsystem.running;
            }
        }

        /// <summary>
        /// Start the real camera even before surveying the marker. Invalid/missing
        /// measurements cannot start image scanning or establish an alignment.
        /// Failure never disables the independent CSI client.
        /// </summary>
        public void Begin(SiteNavigationConfig config)
        {
            if (Ready) { Scan(config); return; }
            if (booting) { SetMessage("AR is starting. You can return to the 2D view at any time."); return; }
            Shutdown();
            string error;
            CaptureMeasurements(config, out error);
            sourceConfig = config;
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!isActiveAndEnabled)
            { SetMessage("AR tracker is disabled. Continue in the 2D view."); return; }
            requested = true; booting = true; paused = false;
            int token = generation;
            SetMessage("Starting the real AR camera. No site alignment is assumed.");
            startupRoutine = StartCoroutine(GuardStartup(StartAndroid(token), token));
#else
            SetMessage("Real site AR requires the Android camera. No editor pose is substituted; use the 2D view.");
#endif
        }

        /// <summary>Discard old calibration and build a fresh measured-size reference-image library.</summary>
        public void Scan(SiteNavigationConfig config)
        {
            if (!Ready)
            {
                if (booting) SetMessage("AR is still starting. The 2D CSI view remains available.");
                else Begin(config);
                return;
            }
            if (paused || imageJobRunning || anchorPending)
            { SetMessage("Wait for the current AR operation to finish, then Scan again."); return; }
            string error;
            if (!CaptureMeasurements(config, out error))
            { InvalidateVerification(error); return; }
            if (!CanPrepareImageLibrary)
            { SetMessage("Waiting for the AR camera. Point it at the room, then tap Scan image again."); return; }
            generation++;
            ClearAlignment(); candidate = null; needsRescan = false;
            images.enabled = false;
            try
            {
                library = images.CreateRuntimeLibrary() as MutableRuntimeReferenceImageLibrary;
                if (library == null) throw new NotSupportedException("The AR provider has no mutable reference-image library.");
                imageJob = library.ScheduleAddImageWithValidationJob(referenceTexture, ReferenceName, imageWidth);
                imageJobGeneration = generation; imageJobRunning = true;
                SetMessage("Validating the reference image at its measured width. Aim at the complete upright wall image.");
            }
            catch (Exception exception)
            {
                InvalidateVerification("Image setup failed: " + exception.Message);
                Debug.LogException(exception);
            }
        }

        /// <summary>
        /// Lock a true AR anchor at the observed image pose. MapRoot is only a
        /// verification-grid frame until the user explicitly calls VerifyAlignment.
        /// </summary>
        public async void LockAlignment()
        {
            if (!HasCandidate || anchors == null || anchors.subsystem == null)
            { SetMessage("Lock requires a currently tracked upright reference image and live camera tracking."); return; }
            string error;
            if (!MeasurementsUnchanged(out error)) { InvalidateVerification(error); return; }
            int token = generation;
            int request = ++anchorRequestId;
            var manager = anchors;
            Pose observedMarker = new Pose(candidate.transform.position, candidate.transform.rotation);
            Pose measuredMarker = markerMapPose;
            ARAnchor created = null;
            anchorPending = true;
            SetMessage("Creating a real image-position AR anchor. Navigation remains unverified.");
            try
            {
                var result = await manager.TryAddAnchorAsync(observedMarker);
                created = result.value;
                if (!result.status.IsSuccess() || created == null)
                    throw new InvalidOperationException("AR anchor rejected: " + result.status);
                // This timeout bounds an anchor operation, never simulated walking.
                float deadline = Time.realtimeSinceStartup + 3f;
                while (Current(token) && created != null && created.trackingState != TrackingState.Tracking
                    && Time.realtimeSinceStartup < deadline) await Awaitable.NextFrameAsync();
                if (!Current(token)) return;
                if (!DeviceTracking || created == null || created.trackingState != TrackingState.Tracking)
                    throw new InvalidOperationException("Anchor/camera is not Tracking. Scan and lock again.");
                if (!UnitScale(created.transform)) throw new InvalidOperationException("AR anchor scale is not 1:1.");
                if (!MeasurementsUnchanged(out error)) throw new InvalidOperationException(error);

                anchor = created; created = null;
                mapRoot = new GameObject("Surveyed site map / metres / verification required").transform;
                mapRoot.SetParent(anchor.transform, false);
                Pose inverseMarker = ComposeMapPose(Pose.identity, measuredMarker);
                mapRoot.localPosition = inverseMarker.position;
                mapRoot.localRotation = inverseMarker.rotation;
                mapRoot.localScale = Vector3.one;
                calibrated = true; verified = false; needsRescan = false;
                // A fixed-image anchor persists while the camera leaves the image.
                // Navigation validity follows the real session and anchor, not image visibility.
                images.enabled = false; candidate = null;
                SetMessage("Alignment locked. Compare real floor points 0, 4 and 10, then confirm.");
                Debug.Log("CSI_SITE_AR_LOCKED: real_anchor=true; scale=1; verified=false; marker_map=" + measuredMarker);
            }
            catch (Exception exception)
            {
                if (Current(token))
                {
                    InvalidateVerification("Alignment lock failed: " + exception.Message);
                    Debug.LogException(exception);
                }
            }
            finally
            {
                RemoveAnchor(created, manager);
                if (request == anchorRequestId)
                { anchorPending = false; Notify(); }
            }
        }

        /// <summary>
        /// The UI must call only after the user confirms comparison with at least
        /// the three physical grid points 0, 4 and 10. This is human confirmation, not
        /// an automatic accuracy measurement or a fabricated sensor observation.
        /// </summary>
        public void VerifyAlignment()
        {
            string error;
            if (!IsCalibrated || IsBusy || !UnitScale(mapRoot))
            { SetMessage("A live 1:1 locked alignment is required before checking floor points 0, 4 and 10."); return; }
            if (!MeasurementsUnchanged(out error)) { InvalidateVerification(error); return; }
            verified = true;
            SetMessage("User confirmed floor points 0, 4 and 10. Tracking loss requires a fresh scan and check.");
            Debug.Log("CSI_SITE_AR_VERIFIED: user_three_point_confirmation=true; measured_accuracy_not_asserted");
        }

        /// <summary>Immediately hide map content and revoke navigation permission.</summary>
        public void InvalidateVerification(string reason)
        {
            generation++;
            calibrated = verified = false; needsRescan = true; candidate = null;
            if (mapRoot != null) mapRoot.gameObject.SetActive(false);
            if (images != null) images.enabled = false;
            SetMessage((string.IsNullOrWhiteSpace(reason) ? "Alignment invalidated." : reason)
                + " Scan, lock and compare real floor points 0, 4 and 10 again. 2D CSI remains available.");
        }

        /// <summary>Rigid transform worldFromMap = worldFromImage * inverse(mapFromImage).</summary>
        public static Pose ComposeMapPose(Pose worldImage, Pose mapImage)
        {
            Quaternion rotation = worldImage.rotation * Quaternion.Inverse(mapImage.rotation);
            return new Pose(worldImage.position - rotation * mapImage.position, rotation);
        }

        private void Update()
        {
            if (imageJobRunning && imageJob.jobHandle.IsCompleted) FinishImageJob();
            if (!Ready || paused) return;
            string error;
            if (!needsRescan && !MeasurementsUnchanged(out error)) { InvalidateVerification(error); return; }
            if ((calibrated || anchorPending) && !IsTracking)
            { InvalidateVerification("AR tracking interrupted: " + ARSession.notTrackingReason + ". Map content is hidden."); return; }
            if (calibrated && (!UnitScale(mapRoot) || !mapRoot.gameObject.activeSelf))
            { InvalidateVerification("The map frame is no longer a visible 1:1 alignment."); return; }

            ARTrackedImage previous = candidate;
            candidate = null;
            string rejected = null;
            if (!needsRescan && !calibrated && !anchorPending && !imageJobRunning
                && DeviceTracking && images != null && images.enabled)
                foreach (var image in images.trackables)
                    if (image.referenceImage.guid == imageGuid)
                    {
                        if (ValidImage(image, out error)) { candidate = image; break; }
                        if (image.trackingState == TrackingState.Tracking) rejected = error;
                    }
            if (candidate != previous)
                SetMessage(candidate != null ? "Reference image Tracking. Lock its measured site alignment."
                    : "Reference image is not currently Tracking. Aim at the complete upright wall image.");
            else if (rejected != null) SetMessage(rejected);
            bool tracking = IsTracking;
            if (tracking != lastTracking) { lastTracking = tracking; Notify(); }
        }

        private bool CaptureMeasurements(SiteNavigationConfig config, out string error)
        {
            if (config == null) { error = "Marker measurements are missing."; return false; }
            if (!config.ValidateMarker(out error)) return false;
            Pose pose = config.MarkerMapPose;
            if (!Finite(pose.position) || !Finite(pose.rotation)
                || Vector3.Dot(pose.rotation * Vector3.forward, Vector3.up) < .999f
                || Mathf.Abs(Vector3.Dot(pose.rotation * Vector3.up, Vector3.up)) > .001f)
            { error = "This registration supports only an upright fixed wall image with its top pointing up."; return false; }
            sourceConfig = config; markerMapPose = pose; imageWidth = config.imageWidthM;
            measuredX = config.imageX; measuredY = config.imageY;
            measuredHeight = config.imageHeightM; measuredFacing = config.imageFacingDeg;
            error = null; return true;
        }

        private bool MeasurementsUnchanged(out string error)
        {
            if (sourceConfig == null || !sourceConfig.ValidateMarker(out error))
            { error = "Measured marker setup is no longer valid."; return false; }
            if (sourceConfig.imageWidthM != imageWidth || sourceConfig.imageX != measuredX
                || sourceConfig.imageY != measuredY || sourceConfig.imageHeightM != measuredHeight
                || sourceConfig.imageFacingDeg != measuredFacing)
            { error = "Marker measurements changed after the scan."; return false; }
            error = null; return true;
        }

        private bool ValidImage(ARTrackedImage image, out string reason)
        {
            reason = null;
            if (image == null || image.trackingState != TrackingState.Tracking || referenceTexture == null)
            { reason = "Reference image is not Tracking."; return false; }
            float expectedHeight = imageWidth * referenceTexture.height / referenceTexture.width;
            if (!Finite(image.size.x) || !Finite(image.size.y) || image.size.x <= 0 || image.size.y <= 0
                || Mathf.Abs(image.size.x - imageWidth) > imageWidth * .15f
                || Mathf.Abs(image.size.y - expectedHeight) > expectedHeight * .15f)
            { reason = "Tracked image dimensions do not match the measured reference size. Recheck image width."; return false; }
            if (!Finite(image.transform.position) || !Finite(image.transform.rotation))
            { reason = "Tracked image pose is not finite."; return false; }
            float minTopDot = Mathf.Cos(MountToleranceDegrees * Mathf.Deg2Rad);
            float maxNormalDot = Mathf.Sin(MountToleranceDegrees * Mathf.Deg2Rad);
            if (Vector3.Dot(image.transform.forward, Vector3.up) < minTopDot
                || Mathf.Abs(Vector3.Dot(image.transform.up, Vector3.up)) > maxNormalDot)
            { reason = "This setup requires an upright fixed wall image, top upward. A flat or inverted image cannot use these site measurements."; return false; }
            return true;
        }

        private void FinishImageJob()
        {
            try
            {
                imageJob.jobHandle.Complete(); imageJobRunning = false;
                if (!Current(imageJobGeneration) || paused) return;
                if (imageJob.status != AddReferenceImageJobStatus.Success || library == null || library.count != 1)
                    throw new InvalidOperationException("Reference image rejected: " + imageJob.status);
                imageGuid = library[0].guid;
                images.referenceLibrary = library; images.enabled = true;
                SetMessage("Scan the complete measured wall image. Wait for Tracking, then Lock alignment.");
            }
            catch (Exception exception)
            {
                imageJobRunning = false;
                if (requested) InvalidateVerification("Reference-image validation failed: " + exception.Message);
                Debug.LogException(exception);
            }
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private IEnumerator StartAndroid(int token)
        {
            // Manual XR initialization requires graphics initialization after Start.
            yield return null;
            if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
            {
                bool answered = false;
                var callbacks = new PermissionCallbacks();
                callbacks.PermissionGranted += _ => answered = true;
                callbacks.PermissionDenied += _ => answered = true;
                callbacks.PermissionDeniedAndDontAskAgain += _ => answered = true;
                SetMessage("Allow camera access for measured site AR, or return to 2D CSI.");
                Permission.RequestUserPermission(Permission.Camera, callbacks);
                while (Current(token) && !answered) yield return null;
            }
            if (!Current(token)) yield break;
            if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
                throw new InvalidOperationException("Camera permission denied. Enable it in Android settings to retry");
            if (UnityEngine.Object.FindFirstObjectByType<ARSession>() != null)
                throw new InvalidOperationException("Another AR session is already active. Exit that AR view first");
            xr = XRGeneralSettings.Instance == null ? null : XRGeneralSettings.Instance.Manager;
            if (xr == null) throw new InvalidOperationException("XR settings are missing");
            if (xr.activeLoader != null)
                throw new InvalidOperationException("Another XR loader is active. Return to the previous view and close AR first");
            ownsLoader = true;
            SetMessage("Initializing the real Android AR provider...");
            yield return xr.InitializeLoader();
            if (!Current(token)) yield break;
            if (xr.activeLoader == null) throw new InvalidOperationException("ARCore provider could not initialize");
            yield return ARSession.CheckAvailability();
            if (!Current(token)) yield break;
            if (ARSession.state == ARSessionState.NeedsInstall)
            { SetMessage("Google Play Services for AR requires installation or update."); yield return ARSession.Install(); }
            if (!Current(token)) yield break;
            if (ARSession.state != ARSessionState.Ready && ARSession.state != ARSessionState.SessionInitializing
                && ARSession.state != ARSessionState.SessionTracking)
                throw new InvalidOperationException("AR is unavailable: " + ARSession.state);
            while (Current(token) && paused) yield return null;
            if (!Current(token)) yield break;
            xr.StartSubsystems();
            referenceTexture = Resources.Load<Texture2D>(ReferenceResource);
            if (referenceTexture == null || !referenceTexture.isReadable)
                throw new InvalidOperationException("Reference image is missing or not CPU-readable");
            CreateRuntime();
            // AR Foundation requires an initializing/tracking session before an
            // image-add job. Creating the hierarchy does not start its camera
            // synchronously. Wait for an actual frame; do not require a pose.
            SetMessage("Starting the camera feed before preparing the reference image...");
            float cameraWait = 0f;
            while (Current(token) && !CanPrepareImageLibrary)
            {
                if (!paused) cameraWait += Time.unscaledDeltaTime;
                if (cameraWait > 15f) throw new InvalidOperationException("AR camera did not deliver a frame. Retry AR camera");
                yield return null;
            }
            if (!Current(token)) yield break;
            ready = true; booting = false;
            startupRoutine = null;
            string measurementError;
            if (sourceConfig != null && CaptureMeasurements(sourceConfig, out measurementError)) Scan(sourceConfig);
            else SetMessage("Real AR camera started, but marker measurements are incomplete. Fill the actual wall-image measurements, then Scan. No map alignment exists.");
        }

        private IEnumerator GuardStartup(IEnumerator startup, int token)
        {
            var pending = new Stack<IEnumerator>(); pending.Push(startup);
            try
            {
                while (Current(token) && pending.Count > 0)
                {
                    IEnumerator current = pending.Peek();
                    object instruction = null; Exception failure = null; bool hasNext = false;
                    try { hasNext = current.MoveNext(); if (hasNext) instruction = current.Current; }
                    catch (Exception exception) { failure = exception; }
                    if (failure != null)
                    {
                        startupRoutine = null;
                        Shutdown();
                        SetMessage("AR initialization failed: " + failure.Message + ". Continue in the 2D CSI view.");
                        Debug.LogException(failure); break;
                    }
                    if (!hasNext) { pending.Pop(); (current as IDisposable)?.Dispose(); }
                    else if (instruction is IEnumerator nested) pending.Push(nested);
                    else yield return instruction;
                }
            }
            finally
            {
                while (pending.Count > 0)
                    try { (pending.Pop() as IDisposable)?.Dispose(); } catch (Exception exception) { Debug.LogException(exception); }
                if (token == generation) { booting = false; startupRoutine = null; Notify(); }
            }
        }
#endif

        private void CreateRuntime()
        {
            // Keep XR independent of any scaled UI hierarchy.
            runtimeRoot = new GameObject("Measured site AR runtime"); runtimeRoot.SetActive(false);
            var sessionObject = new GameObject("Site AR Session", typeof(ARSession), typeof(ARInputManager));
            sessionObject.transform.SetParent(runtimeRoot.transform, false);
            session = sessionObject.GetComponent<ARSession>(); session.attemptUpdate = false;
            var originObject = new GameObject("Site XR Origin"); originObject.SetActive(false);
            originObject.transform.SetParent(runtimeRoot.transform, false);
            var offset = new GameObject("Camera Offset"); offset.transform.SetParent(originObject.transform, false);
            var cameraObject = new GameObject("Site AR Camera", typeof(UnityEngine.Camera));
            cameraObject.transform.SetParent(offset.transform, false); cameraObject.tag = "MainCamera";
            arCamera = cameraObject.GetComponent<UnityEngine.Camera>();
            arCamera.nearClipPlane = .01f; arCamera.farClipPlane = 30;
            arCamera.clearFlags = CameraClearFlags.SolidColor; arCamera.backgroundColor = Color.black;
            var driver = cameraObject.AddComponent<TrackedPoseDriver>();
            driver.SetPoseSource(TrackedPoseDriver.DeviceType.GenericXRDevice, TrackedPoseDriver.TrackedPose.ColorCamera);
            driver.trackingType = TrackedPoseDriver.TrackingType.RotationAndPosition;
            driver.updateType = TrackedPoseDriver.UpdateType.UpdateAndBeforeRender;
            cameraManager = cameraObject.AddComponent<ARCameraManager>();
            cameraManager.requestedFacingDirection = CameraFacingDirection.World; cameraManager.autoFocusRequested = true;
            cameraManager.frameReceived += OnCameraFrame;
            cameraObject.AddComponent<ARCameraBackground>();
            var origin = originObject.AddComponent<XROrigin>();
            origin.Camera = arCamera; origin.CameraFloorOffsetObject = offset;
            origin.CameraYOffset = 0; origin.RequestedTrackingOriginMode = XROrigin.TrackingOriginMode.Device;
            anchors = originObject.AddComponent<ARAnchorManager>();
            images = originObject.AddComponent<ARTrackedImageManager>(); images.enabled = false;
            images.requestedMaxNumberOfMovingImages = 1;
            ARSession.stateChanged += OnSessionState;
            foreach (var camera in UnityEngine.Object.FindObjectsByType<UnityEngine.Camera>(FindObjectsSortMode.None))
                if (camera != arCamera && camera.enabled && camera.targetTexture == null)
                { suspendedCameras.Add(camera); camera.enabled = false; }
            originObject.SetActive(true); runtimeRoot.SetActive(true);
        }

        private void OnCameraFrame(ARCameraFrameEventArgs args) { if (!paused) hasCameraFrame = true; }
        private void OnSessionState(ARSessionStateChangedEventArgs args)
        {
            if (requested && (calibrated || anchorPending) && args.state != ARSessionState.SessionTracking)
                InvalidateVerification("AR session lost Tracking: " + args.state + ". Map content is hidden.");
            Notify();
        }
        private void OnApplicationPause(bool value)
        {
            paused = value; hasCameraFrame = false;
            if (requested && ready && value)
                InvalidateVerification("App entered the background. Alignment has been revoked.");
            // Leave the hierarchy active so ARSession can forward ARCore pause/resume.
            // Returning does not restore verification or expose the hidden map frame.
            if (requested && ready && !value)
                SetMessage("Returned to the app. Scan, lock and verify the measured image alignment again.");
        }

        private void ClearAlignment()
        {
            calibrated = verified = false;
            if (mapRoot != null) { mapRoot.gameObject.SetActive(false); Destroy(mapRoot.gameObject); mapRoot = null; }
            RemoveAnchor(anchor, anchors); anchor = null;
        }
        private static void RemoveAnchor(ARAnchor value, ARAnchorManager manager)
        {
            if (value == null) return;
            try
            {
                if (manager != null && manager.enabled && manager.subsystem != null && manager.TryRemoveAnchor(value)) return;
            }
            catch (Exception exception) { Debug.LogWarning("Site AR anchor cleanup: " + exception.Message); }
            if (value != null) UnityEngine.Object.Destroy(value.gameObject);
        }

        public void Shutdown()
        {
            generation++; anchorRequestId++;
            requested = ready = booting = false; candidate = null; needsRescan = true;
            hasCameraFrame = lastTracking = false;
            if (startupRoutine != null) { StopCoroutine(startupRoutine); startupRoutine = null; }
            if (imageJobRunning)
            {
                try { imageJob.jobHandle.Complete(); } catch (Exception exception) { Debug.LogException(exception); }
                imageJobRunning = false;
            }
            ARSession.stateChanged -= OnSessionState;
            if (cameraManager != null) cameraManager.frameReceived -= OnCameraFrame;
            ClearAlignment();
            if (runtimeRoot != null) { runtimeRoot.SetActive(false); Destroy(runtimeRoot); runtimeRoot = null; }
            if (ownsLoader && xr != null && xr.isInitializationComplete)
            {
                try { xr.StopSubsystems(); xr.DeinitializeLoader(); }
                catch (Exception exception) { Debug.LogException(exception); }
            }
            foreach (var camera in suspendedCameras) if (camera != null) camera.enabled = true;
            suspendedCameras.Clear();
            xr = null; ownsLoader = false; anchors = null; images = null; session = null;
            cameraManager = null; arCamera = null; library = null; referenceTexture = null;
            anchorPending = false;
            SetMessage("Camera AR stopped. Real CSI remains available in the independent 2D view.");
        }

        private bool Current(int token) { return this != null && requested && token == generation; }
        private static bool Finite(float value) { return !float.IsNaN(value) && !float.IsInfinity(value); }
        private static bool Finite(Vector3 value) { return Finite(value.x) && Finite(value.y) && Finite(value.z); }
        private static bool Finite(Quaternion value)
        { return Finite(value.x) && Finite(value.y) && Finite(value.z) && Finite(value.w) && Quaternion.Dot(value, value) > .001f; }
        private static bool UnitScale(Transform value)
        { return value != null && (value.lossyScale - Vector3.one).sqrMagnitude < .000001f; }
        private void SetMessage(string value) { Message = value; Notify(); }
        private void Notify() { var callback = Changed; if (callback != null) callback(); }
        private void OnDisable() { Shutdown(); }
        private void OnDestroy() { Shutdown(); }
    }
}
