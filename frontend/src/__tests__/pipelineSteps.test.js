// The JS twin of backend/tests/test_pipeline_steps.py.
//
// That Python test exists because a step key with no renderer is a KeyError on
// a screen nobody opens until a run reaches that stage — it fails quietly, late,
// and only for the person whose cohort got that far. The JS port had no
// equivalent, and drifted: the Slurm in-flight branches for Stages 5 and 6 were
// missing, so a run whose registration job was actively writing displayed
// "ready to register", which invites a second write of a whole cohort's
// identity rows.
//
// So these assert the two things that drift: that every step key has a
// renderer, and that a Slurm-backed write in flight is never reported as ready.
import { describe, expect, it } from "vitest";

import { pipelineSteps, SLURM_IN_FLIGHT } from "../components/pipeline/utils.js";
import { STAGE_RENDERERS } from "../components/pipeline/stages/index.js";

const KEYS = ["tiling", "packaging", "extraction", "assignment", "registration", "kb_load"];

/** A status where everything up to and including Stage 4 is finished, so
 * Stages 5 and 6 are the ones under test rather than blocked behind others. */
function readyForKb(extra = {}) {
  return {
    tiling_total_slides: 10,
    tiling_tiled_slides: 10,
    h5_ready: true,
    extraction_ready: true,
    assignment_ready: true,
    registration_ready: true,
    registration_done: true,
    ...extra,
  };
}

function step(status, key) {
  return pipelineSteps(status).find((s) => s.key === key);
}

describe("pipelineSteps", () => {
  it("returns all six stages, in order, with the expected keys", () => {
    const steps = pipelineSteps({});
    expect(steps.map((s) => s.key)).toEqual(KEYS);
  });

  it("gives every step a renderer", () => {
    // The failure this mirrors from the Python side: a key with no renderer
    // throws only once a run reaches that stage.
    for (const s of pipelineSteps({})) {
      expect(STAGE_RENDERERS[s.key], `no renderer for step "${s.key}"`).toBeTruthy();
    }
  });

  it("gives every step a state and a summary", () => {
    for (const s of pipelineSteps({})) {
      expect(typeof s.state).toBe("string");
      expect(s.state.length).toBeGreaterThan(0);
      expect(typeof s.summary).toBe("string");
    }
  });

  describe("a Slurm-backed KB write in flight", () => {
    // The whole reason this file exists. Parameterised over every in-flight
    // state, because the bug was one missing branch and a test naming only
    // RUNNING would have passed while PENDING still read as ready.
    for (const slurmState of SLURM_IN_FLIGHT) {
      it(`registration is "running", not "action", while ${slurmState}`, () => {
        const s = step(
          readyForKb({ registration_done: false, registration_job_id: "123", registration_slurm_state: slurmState }),
          "registration"
        );
        expect(s.state).toBe("running");
        expect(s.summary).toContain(slurmState);
      });

      it(`kb_load is "running", not "action", while ${slurmState}`, () => {
        const s = step(
          readyForKb({ kb_load_job_id: "456", kb_load_slurm_state: slurmState }),
          "kb_load"
        );
        expect(s.state).toBe("running");
        expect(s.summary).toContain(slurmState);
      });
    }
  });

  it("reports a registration job that ended without committing", () => {
    const s = step(
      readyForKb({ registration_done: false, registration_job_id: "123", registration_slurm_state: "FAILED" }),
      "registration"
    );
    expect(s.state).toBe("attention");
  });

  it("reports a kb_load job that ended without committing", () => {
    const s = step(readyForKb({ kb_load_job_id: "456", kb_load_slurm_state: "FAILED" }), "kb_load");
    expect(s.state).toBe("attention");
  });

  it("still offers registration when no job has ever been submitted", () => {
    // The guards above must not swallow the ordinary case.
    const s = step(readyForKb({ registration_done: false }), "registration");
    expect(s.state).toBe("action");
  });

  it("still offers the KB load when no job has ever been submitted", () => {
    expect(step(readyForKb(), "kb_load").state).toBe("action");
  });

  it("prefers done over an in-flight state", () => {
    // A job id left over from an earlier attempt must not un-complete a stage
    // that has recorded a commit.
    const s = step(
      readyForKb({ registration_job_id: "123", registration_slurm_state: "RUNNING" }),
      "registration"
    );
    expect(s.state).toBe("done");
  });

  it("keeps the KB load blocked behind registration", () => {
    const s = step(readyForKb({ registration_done: false }), "kb_load");
    expect(s.state).toBe("blocked");
    expect(s.summary).toContain("registration");
  });
});
