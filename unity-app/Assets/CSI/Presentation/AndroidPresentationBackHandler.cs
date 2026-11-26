using System;
using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using System.Collections.Concurrent;
using System.Threading;
using UnityEngine.Scripting;
#endif

namespace CsiApp
{
    /// <summary>Owns Android Back only while the independent model page is active.</summary>
    internal sealed class AndroidPresentationBackHandler : IDisposable
    {
        private bool disposed;
#if UNITY_ANDROID && !UNITY_EDITOR
        private readonly bool previousBackPolicy;
        private readonly BackRegistration registration;
#endif

        public AndroidPresentationBackHandler()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            previousBackPolicy = Input.backButtonLeavesApp;
            Input.backButtonLeavesApp = false;
            try
            {
                using (var version = new AndroidJavaClass("android.os.Build$VERSION"))
                {
                    int sdk = version.GetStatic<int>("SDK_INT");
                    if (sdk >= 33)
                    {
                        registration = new BackRegistration();
                        PostToAndroidUi(registration.RegisterOnUiThread);
                    }
                    Debug.Log("CSI_MOBILE_MODEL_BACK_POLICY: sdk=" + sdk + "; leavesApp=false; previous=" + previousBackPolicy);
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning("CSI_MOBILE_MODEL_BACK_SETUP_FAILED: " + exception.Message + "; Escape fallback remains active.");
            }
#endif
        }

        /// <summary>Called only from the Unity player loop; never loads a scene on the Java thread.</summary>
        public bool ConsumeBackRequest()
        {
            if (disposed) return false;
#if UNITY_ANDROID && !UNITY_EDITOR
            if (registration != null)
            {
                string diagnostic;
                while (registration.Diagnostics.TryDequeue(out diagnostic)) Debug.Log(diagnostic);
                if (registration.Consume())
                {
                    Debug.Log("CSI_MOBILE_MODEL_BACK_INVOKED: Android predictive Back -> Unity Update");
                    return true;
                }
            }
#endif
            return false;
        }

        public void DiscardPending()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (registration != null) registration.DiscardPending();
#endif
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
#if UNITY_ANDROID && !UNITY_EDITOR
            // Invalidate immediately, before asynchronous JNI cleanup or scene destruction.
            if (registration != null)
            {
                registration.Close();
                try { PostToAndroidUi(registration.UnregisterOnUiThread); }
                catch (Exception exception)
                {
                    Debug.LogWarning("CSI_MOBILE_MODEL_BACK_RELEASE_FAILED: " + exception.Message);
                }
            }
            Input.backButtonLeavesApp = previousBackPolicy;
            Debug.Log("CSI_MOBILE_MODEL_BACK_RELEASE: callback invalidated; leavesApp restored=" + previousBackPolicy);
#endif
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private static void PostToAndroidUi(Action action)
        {
            // The Java Runnable retains its managed delegate until the queued UI task runs.
            using (var unity = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (var activity = unity.GetStatic<AndroidJavaObject>("currentActivity"))
                activity.Call("runOnUiThread", new AndroidJavaRunnable(() => action()));
        }

        /// <summary>Plain managed state: no reference to a MonoBehaviour or any scene object.</summary>
        private sealed class BackRegistration
        {
            public readonly ConcurrentQueue<string> Diagnostics = new ConcurrentQueue<string>();
            private int closed, pending;
            // JNI objects below are created, accessed and released only on the Android UI thread.
            private AndroidJavaObject dispatcher;
            private BackCallback callback;
            private bool registered;

            public void RegisterOnUiThread()
            {
                if (Volatile.Read(ref closed) != 0) return;
                try
                {
                    using (var unity = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                    using (var activity = unity.GetStatic<AndroidJavaObject>("currentActivity"))
                        dispatcher = activity.Call<AndroidJavaObject>("getOnBackInvokedDispatcher");
                    callback = new BackCallback(this); // Keep a strong reference until unregister completes.
                    dispatcher.Call("registerOnBackInvokedCallback", 0, callback); // PRIORITY_DEFAULT
                    registered = true;
                    if (Volatile.Read(ref closed) != 0) UnregisterOnUiThread();
                    else Diagnostics.Enqueue("CSI_MOBILE_MODEL_BACK_REGISTERED: OnBackInvokedCallback; priority=0");
                }
                catch (Exception exception)
                {
                    Diagnostics.Enqueue("CSI_MOBILE_MODEL_BACK_REGISTER_FAILED: " + exception.Message);
                    UnregisterOnUiThread();
                }
            }

            public void Signal()
            {
                if (Volatile.Read(ref closed) == 0) Interlocked.Exchange(ref pending, 1);
            }

            public bool Consume()
            {
                return Interlocked.Exchange(ref pending, 0) != 0 && Volatile.Read(ref closed) == 0;
            }

            public void DiscardPending() { Interlocked.Exchange(ref pending, 0); }
            public void Close() { Interlocked.Exchange(ref closed, 1); DiscardPending(); }

            public void UnregisterOnUiThread()
            {
                try
                {
                    if (registered && dispatcher != null && callback != null)
                        dispatcher.Call("unregisterOnBackInvokedCallback", callback);
                }
                catch (Exception exception)
                {
                    // A closed callback is harmless even if Android is already destroying its window.
                    Diagnostics.Enqueue("CSI_MOBILE_MODEL_BACK_UNREGISTER_FAILED: " + exception.Message);
                }
                finally
                {
                    registered = false;
                    if (dispatcher != null) dispatcher.Dispose();
                    dispatcher = null;
                    callback = null;
                }
            }
        }

        [Preserve]
        private sealed class BackCallback : AndroidJavaProxy
        {
            private readonly BackRegistration state;
            public BackCallback(BackRegistration state) : base("android.window.OnBackInvokedCallback") { this.state = state; }
            [Preserve]
            public void onBackInvoked() { state.Signal(); }
        }
#endif
    }
}
