import { useEffect, useRef, useState, useCallback } from "react";

// JS analogue of Streamlit's @st.fragment(run_every="Ns"): re-runs `fetcher`
// on an interval and exposes the latest result, loading state, and error.
// `enabled` lets a caller pause polling (e.g. while a step isn't the active
// one), and `deps` restarts the interval when identity (submission id, etc.)
// changes.
export function usePolling(fetcher, { intervalMs, enabled = true, deps = [] }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refresh = useCallback(async () => {
    try {
      const result = await fetcherRef.current();
      setData(result);
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    setLoading(true);
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs, refresh, ...deps]);

  return { data, error, loading, refresh };
}
