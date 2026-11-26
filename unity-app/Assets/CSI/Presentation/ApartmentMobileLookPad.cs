using UnityEngine;
using UnityEngine.EventSystems;

namespace CsiApp
{
    /// <summary>Only a gesture that began on the model surface can rotate the virtual view.</summary>
    public sealed class ApartmentMobileLookPad : MonoBehaviour, IPointerDownHandler, IPointerUpHandler, IBeginDragHandler, IDragHandler, IEndDragHandler
    {
        public ApartmentPresentationController Controller;
        private int pointer = int.MinValue;
        public void OnPointerDown(PointerEventData data)
        {
            if (pointer != int.MinValue || data.pointerPressRaycast.gameObject != gameObject) return;
            pointer = data.pointerId;
        }
        public void OnBeginDrag(PointerEventData data) { }
        public void OnDrag(PointerEventData data)
        {
            if (data.pointerId != pointer || Controller == null) return;
            Controller.RotateVirtualView(data.delta.x / Mathf.Max(1, Screen.width) * 180,
                -data.delta.y / Mathf.Max(1, Screen.width) * 140);
        }
        public void OnPointerUp(PointerEventData data) { if (data.pointerId == pointer) Cancel(); }
        public void OnEndDrag(PointerEventData data) { if (data.pointerId == pointer) Cancel(); }
        public void Cancel() { pointer = int.MinValue; }
        private void OnDisable() { Cancel(); }
    }
}
