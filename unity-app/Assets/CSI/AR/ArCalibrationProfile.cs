using UnityEngine;

namespace CsiApp
{
    // The marker's measured pose in apartment coordinates has not been supplied.
    // A lab anchor is deliberately not a site calibration or a CSI position.
    public sealed class ArCalibrationProfile
    {
        public const float LabModelScale = .05f;
        public float ImageWidthMeters = .20f;
        public Pose? ApartmentImageWorldPose = null;
        public bool HasSiteAlignment { get { return ApartmentImageWorldPose.HasValue; } }
    }

    // Meeting-only placement choices. None of these values establish a measured site pose.
    public enum ArMeetingView { Miniature, FullSizeOverlay }

    public static class ArMeetingPlacement
    {
        public static void Place(Transform root, Transform markerAnchor, Vector3 cameraForward,
            ArMeetingView view, float illustrativeMarkerHeight)
        {
            Vector3 towardScanner = -Vector3.ProjectOnPlane(cameraForward, Vector3.up);
            if (towardScanner.sqrMagnitude < .01f)
                throw new System.InvalidOperationException("Hold the phone more upright before locking the demo.");
            Quaternion horizontal = Quaternion.LookRotation(towardScanner.normalized, Vector3.up);
            float scale = view == ArMeetingView.Miniature ? ArCalibrationProfile.LabModelScale : 1f;
            Vector3 position = markerAnchor.position;
            if (view == ArMeetingView.Miniature)
            {
                Vector2 center = ApartmentMap.Current.PlanBounds.center;
                position -= horizontal * ApartmentMap.ToWorld(center.x, center.y) * scale;
            }
            else position -= Vector3.up * illustrativeMarkerHeight;
            root.SetParent(markerAnchor, false);
            root.localScale = Vector3.one * scale;
            root.SetPositionAndRotation(position, horizontal);
        }
    }
}
