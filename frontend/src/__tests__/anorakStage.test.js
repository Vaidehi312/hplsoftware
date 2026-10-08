// The React side of backend/tests/test_anorak_server.py.
//
// Stage 7's form used to be reachable while the head job was alive: the form
// sent overwrite on every click on the grounds that "the in-flight check is the
// server's", and the server skipped that check whenever overwrite was set. The
// states that let the form show — Slurm unreachable, CONFIGURING — are asserted
// here against the real stepper and the real component, rendered to a string.
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api.js";
import AnorakStage from "../components/pipeline/stages/AnorakStage.jsx";
import { AnorakRunStage } from "../components/pipeline/stages/PipelineStage.jsx";
import { pipelineSteps, SLURM_IN_FLIGHT } from "../components/pipeline/utils.js";

function anorakStep(status) {
  return pipelineSteps(status).find((s) => s.key === "anorak");
}

function render(status) {
  return renderToStaticMarkup(
    createElement(AnorakStage, { status, submissionId: "sub1", onChanged: () => {} })
  );
}

describe("ANORAK stage state", () => {
  it("counts CONFIGURING as in flight, as the server does", () => {
    expect(SLURM_IN_FLIGHT.has("CONFIGURING")).toBe(true);
    const status = { anorak_job_id: "1111", anorak_slurm_state: "CONFIGURING" };
    expect(anorakStep(status).state).toBe("running");
    expect(render(status)).not.toContain("Retry ANORAK");
  });

  it("does not call an unknown state 'did not finish', or offer a retry over it", () => {
    const status = {
      anorak_job_id: "1111",
      anorak_slurm_state: null,
      anorak_state_unknown: true,
      anorak_submit_blocked: "Couldn't reach Slurm to confirm ANORAK job 1111 has stopped",
    };
    expect(anorakStep(status).summary).toContain("unknown");
    const html = render(status);
    expect(html).toContain("reach Slurm");
    expect(html).not.toContain("Retry ANORAK");
  });

  it("takes the server's in-flight verdict over its own copy of the states", () => {
    const status = { anorak_job_id: "1111", anorak_slurm_state: "SOME_NEW_STATE", anorak_in_flight: true };
    expect(anorakStep(status).state).toBe("running");
    expect(render(status)).not.toContain("Retry ANORAK");
  });

  it("still offers a retry once the head job has stopped", () => {
    const status = { anorak_job_id: "1111", anorak_slurm_state: "FAILED", anorak_state_unknown: false };
    expect(render(status)).toContain("Retry ANORAK");
  });

  it("shows a recorded submission error", () => {
    expect(render({ anorak_error: "sbatch failed (exit 1): invalid partition" })).toContain(
      "invalid partition"
    );
  });
});

describe("ANORAK stage: what a stopped run and a new one say", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows why a run stopped for good, from the supervisor's marker", () => {
    // A bare FAILED was all the stage said; the reason sat in nf_supervise.stop.
    const html = render({
      anorak_job_id: "1111,1112",
      anorak_slurm_state: "FAILED",
      anorak_state_unknown: false,
      anorak_stop_reason: "nextflow exited 1 on its own: a pipeline failure no restart can fix",
    });
    expect(html).toContain("stop marker");
    expect(html).toContain("a pipeline failure no restart can fix");
  });

  it("says a sample is required on every row, not optional", () => {
    // Blank samples used to be pooled into one tumour; the list is now refused.
    const html = render({});
    expect(html).not.toContain("if you want grades aggregated per tumour");
    expect(html).toContain("filled in on every row");
    expect(html).toContain("Head jobs");
  });

  it("submits a standby head job by default", async () => {
    // Without one, a head job that reaches its walltime ends the run.
    const calls = [];
    vi.stubGlobal("fetch", async (url, init) => {
      calls.push(JSON.parse(init.body));
      return new Response(JSON.stringify({ selection: {} }), { status: 200 });
    });
    await api.startAnorak("sub1", { slidesCsv: "/x.csv" });
    expect(calls[0].chain).toBe(2);
  });
});


// The live run of 2026-09-30: the recorded head job (1255076) reached its
// walltime and a chain submitted by hand under the run's name carried on. The
// stage showed TIMEOUT and a Retry form over a running pipeline.
function liveStatus(overrides = {}) {
  return {
    anorak_job_id: "1255076",
    anorak_slurm_state: "RUNNING",
    anorak_in_flight: true,
    anorak_state_unknown: false,
    anorak_submit_blocked: "ANORAK is already running for this run",
    anorak_recorded_state: "TIMEOUT",
    anorak_chain_took_over: true,
    anorak_head_job_id: "1264266",
    anorak_head_state: "RUNNING",
    anorak_head_time_left: "1-10:21:07",
    anorak_standby_job_ids: ["1264267", "1264268", "1264269", "1264270", "1264271", "1264272"],
    anorak_standbys_queued: 6,
    anorak_progress: {
      slides: 7221,
      counts_source: "squeue",
      trace_available: true,
      stitched_slides: 192,
      steps: [
        { process: "TILE_SLIDE", done: 7221, running: 0, waiting: 0, failed: 0, total: 7221 },
        { process: "PREDICT_GP", done: 300, running: 8, waiting: 192, failed: 2, total: 7221 },
        { process: "TUMOUR_GRADE", done: 0, running: 0, waiting: 0, failed: 0, total: 1 },
      ],
      task_queue: {
        jobs: 200,
        waiting: [
          { reason: "AssocGrpGRES", count: 192, explanation: "your Slurm account's GPU limit is in use" },
        ],
      },
    },
    ...overrides,
  };
}

function renderRun(status) {
  return renderToStaticMarkup(
    createElement(AnorakRunStage, { status, submissionId: "sub1", onChanged: () => {} })
  );
}

describe("ANORAK stage: the live head chain and progress", () => {
  for (const [name, draw] of [["Stage 7", render], ["an ANORAK run", renderRun]]) {
    it(`${name}: shows the live head job, the takeover and the standbys, not a retry`, () => {
      const html = draw(liveStatus());
      expect(html).toContain("Head job 1264266");
      expect(html).toContain("1-10:21:07 left");
      expect(html).toContain("ended as TIMEOUT; the chain took over");
      expect(html).toContain("6 standby head jobs queued");
      expect(html).toContain("the next standby resumes the run; finished slides are kept");
      expect(html).not.toContain("nothing will take over");
      expect(html).not.toContain("Retry ANORAK");
      expect(html).not.toContain("Resume ANORAK");
    });

    it(`${name}: shows per-step progress, stitched slides and why tasks wait`, () => {
      const html = draw(liveStatus());
      expect(html).toContain("Predict growth patterns (GPU): 300 / 7,221 done · 8 running, 192 waiting, 2 failed");
      expect(html).toContain("Tile slides: 7,221 / 7,221 done");
      expect(html).toContain("Tumour grades: 0 / 1 done");
      expect(html).toContain("Stitched slides in ss1_final: 192 of 7,221");
      expect(html).toContain("GPU limit is in use");
      expect(html).toContain("AssocGrpGRES");
    });

    it(`${name}: warns when no standby will take over at the time limit`, () => {
      const html = draw(
        liveStatus({ anorak_chain_took_over: false, anorak_standby_job_ids: [], anorak_standbys_queued: 0 })
      );
      expect(html).toContain("No standby head job is queued: nothing will take over when head job 1264266");
      expect(html).toContain("(in 1-10:21:07)");
      expect(html).not.toContain("the chain took over");
    });
  }

  it("keeps the stopped view, with how far it got", () => {
    const html = render(
      liveStatus({
        anorak_slurm_state: "TIMEOUT",
        anorak_in_flight: false,
        anorak_submit_blocked: null,
        anorak_chain_took_over: false,
        anorak_head_job_id: null,
        anorak_standby_job_ids: [],
      })
    );
    expect(html).toContain("This attempt ended as: TIMEOUT");
    expect(html).toContain("Retry ANORAK");
    expect(html).toContain("300 / 7,221 done");
    expect(html).not.toContain("Head job 1264266");
  });
});
