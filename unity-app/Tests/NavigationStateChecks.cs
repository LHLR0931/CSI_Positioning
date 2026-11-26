using System;
using CsiApp;

public static class NavigationStateChecks
{
    private static int assertions;
    private static void Check(bool result, string name) { assertions++; if (!result) throw new Exception(name); }
    private static PositionSample Sample(int n, long seq) { return new PositionSample { Label = n, Sequence = seq, ReceivedUtc = DateTime.UtcNow }; }
    public static void Main()
    {
        for (int a = 0; a < 15; a++) for (int b = 0; b < 15; b++)
        {
            var path = GridMap.Route(a, b);
            Check(path[0] == a && path[path.Count - 1] == b, "Endpoints");
            Check(path.Count - 1 == Math.Abs(a % 5 - b % 5) + Math.Abs(a / 5 - b / 5), "Shortest four-neighbour route");
            for (int i = 1; i < path.Count; i++) Check(GridMap.Neighbours(path[i - 1]).Contains(path[i]), "No diagonal or row wrap");
        }
        bool rejected = false; try { GridMap.Label(5, 0); } catch (ArgumentOutOfRangeException) { rejected = true; }
        Check(rejected, "Out of range");
        var s = new NavigationState();
        Check(s.TopView && !s.Balcony && s.Target == 14, "Default is a top-view grid target, not a guessed balcony exit");
        Check(s.DestinationAvailable && s.Path[0] == 12 && s.Path[s.Path.Count - 1] == 14, "Default grid preview is ready");
        Check(Math.Abs(s.PlannedGridLength - 1.6) < 1e-9, "Default grid route is two measured grid edges");
        Check(s.Start(), "Default grid demo starts without simulated AR calibration");
        while (s.Path.Count > 1) s.DemoSample();
        Check(s.Navigating && s.Prediction == 14, "Reaching a grid target is not automatic arrival");
        s.SetOutside(true); int last = s.Prediction; s.Accept(Sample(14, s.Sequence + 1));
        Check(s.Prediction == last && s.LastRaw.Label == 14, "Freeze marker while preserving raw sample");
        s.SetOutside(false); s.PauseOrResume(); Check(s.Paused, "Re-entry requires a new sample");
        s.Accept(Sample(14, s.Sequence + 1)); Check(s.Paused, "New data does not auto resume");
        s.PauseOrResume(); Check(!s.Paused && s.Path[0] == 14, "Resume rebuilds route from current estimate");
        s.SetStale(true); Check(s.Paused, "Stale pauses navigation");
        s.Accept(Sample(4, s.Sequence + 1)); Check(s.Paused, "Fresh result keeps explicit pause");
        Check(!s.Accept(Sample(9, s.Sequence)), "Duplicate result ignored");
        Check(!s.Accept(new PositionSample { Label = 9, Sequence = s.Sequence + 1, Source = "Live" }), "No disguised live adapter");
        s.End(true); s.Accept(Sample(3, s.Sequence + 1)); Check(!s.Navigating, "Ended navigation stays ended");
        s.StartRecording(); s.Accept(Sample(3, s.Sequence + 1));
        Check(s.RecordCount == 1 && s.Errors.Count == 0 && double.IsNaN(s.MeanError), "Missing truth is N/A");
        s.GroundTruth = 4; s.Accept(Sample(3, s.Sequence + 1));
        Check(Math.Abs(s.MeanError - .8) < 1e-9, "Metre error");
        s.StopRecording(); Check(s.GroundTruth == -1, "Stop clears truth interval");
        Check(s.Csv().Contains(",demo,") && s.Csv().Contains("error_m"), "Demo provenance exported");

        var pending = new NavigationState();
        Check(pending.Start(), "Start a grid route before switching to an unavailable destination");
        pending.SetTarget(true, 5);
        Check(pending.Balcony && pending.Target == -1 && !pending.DestinationAvailable, "Legacy label 5 cannot become an unconfirmed balcony exit");
        Check(!pending.Navigating && pending.Path.Count == 0 && double.IsNaN(pending.PlannedGridLength), "Unconfigured balcony clears the old route and length");
        Check(!pending.Start() && !pending.CanResume, "Unconfigured balcony cannot start or resume");
        pending.TopView = false;
        Check(!pending.Start(), "3D preview does not make the balcony destination available");
        pending.DemoSample();
        Check(pending.Path.Count == 0 && !pending.Navigating, "Fresh demo position cannot create a balcony route");
        pending.SetTarget(false, 0);
        Check(pending.DestinationAvailable && pending.Path[pending.Path.Count - 1] == 0, "Grid targets remain usable after a pending destination");
        Check(pending.Start(), "Virtual 3D grid navigation does not require fake calibration");
        pending.InvalidateAr();
        Check(pending.Paused && !pending.CanResume, "Simulated tracking loss pauses demo navigation");
        pending.TopView = true; pending.PauseOrResume();
        Check(pending.Paused, "Changing viewpoint does not resolve a simulated tracking fault");
        pending.RestoreTrackingDemo();
        Check(pending.Paused && pending.CanResume, "Restoring a simulated fault still requires explicit resume");
        pending.PauseOrResume(); Check(!pending.Paused, "Explicit resume follows fault recovery");
        pending.SetStale(true); pending.InvalidateAr(); pending.RestoreTrackingDemo();
        pending.PauseOrResume(); Check(pending.Paused, "Resolving tracking alone does not clear stale CSI");
        pending.Accept(Sample(10, pending.Sequence + 1));
        Check(pending.Paused && pending.CanResume, "Fresh data after multiple faults does not auto-resume");
        pending.PauseOrResume();
        Check(!pending.Paused && pending.Path[0] == 10, "Resumed route starts at latest accepted grid point");
        pending.End(false); pending.RestoreTrackingDemo(); pending.DemoSample();
        Check(!pending.Navigating, "Recovery and data do not restart ended navigation");
        Console.WriteLine("PASS: " + assertions + " assertions; 225 grid pairs, pending destinations, both virtual views and demo failure states.");
    }
}
