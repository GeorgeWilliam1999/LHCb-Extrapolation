/* RK-PINN Model Map: architecture pop-up diagrams, one per model type.
 *
 * Usage on the page:
 *   <link rel="stylesheet" href="architectures.css">   (or inline it)
 *   <script src="architectures.js"></script>
 *   const t = RKPINN_ARCH.typeOf(configObject);       // parsed configuration.yaml
 *   const html = RKPINN_ARCH.render(t, configObject);  // figure(s) + ordered steps, as an HTML string
 *
 * Every box is checked against the rkpinn package (networks/, losses/, training/training_protocols.py)
 * and the 68 experiment configurations on main (14 Nov 2026).
 */
(function () {
  "use strict";

  // ---------- small SVG builder: boxes on a grid, labelled orthogonal arrows ----------
  var CW = 196, BW = 152, BH = 58, RH = 104, PADX = 16, PADY = 34;

  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  function box(n) {
    var x = PADX + n.c * CW, y = PADY + n.r * RH, w = n.w ? n.w * CW - (CW - BW) : BW;
    var lines = (n.sub || "").split("\n").filter(Boolean);
    var cls = "ak-" + (n.k || "state");
    var dash = n.k === "use" ? ' stroke-dasharray="5 4"' : "";
    var t = '<g class="ak-node ' + cls + '">' +
      '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + BH + '" rx="7" fill="var(--ak-fill)" stroke="var(--ak-stroke)" stroke-width="1.4"' + dash + '></rect>' +
      (n.num ? '<circle cx="' + (x + 2) + '" cy="' + (y + 2) + '" r="9" fill="var(--ak-numbg)" stroke="var(--ak-stroke)" stroke-width="1"></circle><text x="' + (x + 2) + '" y="' + (y + 5.5) + '" text-anchor="middle" class="ak-num" fill="var(--ak-numfg)">' + n.num + "</text>" : "") +
      '<text x="' + (x + w / 2) + '" y="' + (y + (lines.length ? 21 : 33)) + '" text-anchor="middle" class="ak-t" fill="var(--ak-ink)">' + esc(n.t) + "</text>";
    lines.forEach(function (l, i) {
      t += '<text x="' + (x + w / 2) + '" y="' + (y + 36 + i * 13) + '" text-anchor="middle" class="ak-s" fill="var(--ak-sub)">' + esc(l) + "</text>";
    });
    return t + "</g>";
  }

  function anchor(n, side) {
    var x = PADX + n.c * CW, y = PADY + n.r * RH, w = n.w ? n.w * CW - (CW - BW) : BW;
    if (side === "r") return [x + w, y + BH / 2];
    if (side === "l") return [x, y + BH / 2];
    if (side === "b") return [x + w / 2, y + BH];
    if (side === "t") return [x + w / 2, y];
  }

  function edge(nodes, e, uid) {
    var a = nodes[e[0]], b = nodes[e[1]], lab = e[2] || "", style = e[3] || "";
    var cls = style.indexOf("grad") >= 0 ? "ak-e ak-grad" : style.indexOf("train") >= 0 ? "ak-e ak-train" : "ak-e";
    var dash = style.indexOf("train") >= 0 || style.indexOf("grad") >= 0 ? ' stroke-dasharray="5 4"' : "";
    var p, lx, ly;
    if (style.indexOf("loop") >= 0) {             // back along the top, from b-top to a-top (chain to next step)
      var s = anchor(a, "t"), d = anchor(b, "t"), top = PADY - 22;
      p = "M" + s[0] + "," + s[1] + " V" + top + " H" + d[0] + " V" + (d[1] - 2);
      lx = (s[0] + d[0]) / 2; ly = top - 4;
    } else if (a.r === b.r && b.c > a.c) {
      var s1 = anchor(a, "r"), d1 = anchor(b, "l");
      p = "M" + s1[0] + "," + s1[1] + " H" + (d1[0] - 2);
      lx = (s1[0] + d1[0]) / 2; ly = s1[1] - 6;
    } else if (a.r === b.r && b.c < a.c && style.indexOf("below") < 0) {
      var s5 = anchor(a, "l"), d5 = anchor(b, "r");
      p = "M" + s5[0] + "," + s5[1] + " H" + (d5[0] + 2);
      lx = (s5[0] + d5[0]) / 2; ly = s5[1] - 6;
    } else if (a.r === b.r && b.c < a.c) {
      var s2 = anchor(a, "b"), d2 = anchor(b, "b"), low = s2[1] + 18;
      p = "M" + s2[0] + "," + s2[1] + " V" + low + " H" + d2[0] + " V" + (d2[1] + 2);
      lx = (s2[0] + d2[0]) / 2; ly = low + 12;
    } else if (b.r > a.r) {
      var s3 = anchor(a, "b"), d3 = anchor(b, "t"), mid = s3[1] + (d3[1] - s3[1]) / 2;
      p = "M" + s3[0] + "," + s3[1] + " V" + mid + " H" + d3[0] + " V" + (d3[1] - 2);
      lx = s3[0] === d3[0] ? s3[0] + 6 : (s3[0] + d3[0]) / 2; ly = mid - 5;
    } else {
      var s4 = anchor(a, "t"), d4 = anchor(b, "b"), mid2 = d4[1] + (s4[1] - d4[1]) / 2;
      p = "M" + s4[0] + "," + s4[1] + " V" + mid2 + " H" + d4[0] + " V" + (d4[1] + 2);
      lx = s4[0] === d4[0] ? s4[0] + 6 : (s4[0] + d4[0]) / 2; ly = mid2 - 5;
    }
    var anchorMode = "middle";
    if (p.indexOf(" H") < 0 || /V[\d.]+ H([\d.]+) V/.test(p) && (function () { var m = p.match(/^M([\d.]+),/); var h = p.match(/ H([\d.]+) V/); return m && h && Math.abs(+m[1] - +h[1]) < 0.5; })()) { anchorMode = "start"; }
    return '<path d="' + p + '" class="' + cls + '" fill="none" stroke="var(--ak-edge)" stroke-width="1.5"' + dash +
      ' marker-end="url(#ak-arr-' + uid + ')"></path>' +
      (lab ? '<text x="' + lx + '" y="' + ly + '" text-anchor="' + anchorMode + '" class="ak-el" fill="var(--ak-sub)">' + esc(lab) + "</text>" : "");
  }

  var uidc = 0;
  function svg(spec, aria) {
    var uid = "a" + (++uidc), nodes = {}, maxc = 0, maxr = 0;
    spec.nodes.forEach(function (n) { nodes[n.id] = n; maxc = Math.max(maxc, n.c + (n.w || 1)); maxr = Math.max(maxr, n.r); });
    var W = PADX * 2 + maxc * CW - (CW - BW), H = PADY + (maxr + 1) * RH - (RH - BH) + 30;
    var out = '<svg viewBox="0 0 ' + W + " " + H + '" style="max-width:' + W + 'px" role="img" aria-label="' + esc(aria) + '" class="ak-svg">' +
      '<defs><marker id="ak-arr-' + uid + '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
      '<path d="M0,0 L10,5 L0,10 z" fill="var(--ak-edge)"></path></marker></defs>';
    spec.edges.forEach(function (e) { out += edge(nodes, e, uid); });
    spec.nodes.forEach(function (n) { out += box(n); });
    return out + "</svg>";
  }

  // ---------- helpers that read a configuration ----------
  function g(o, path, d) { var v = o; path.split(".").forEach(function (k) { v = (v && v[k] !== undefined) ? v[k] : undefined; }); return v === undefined ? d : v; }
  function q(cfg) { return g(cfg, "steps.number_of_stages", 2); }
  function N(cfg) { return g(cfg, "steps.number_of_steps", 64); }
  function W(cfg, d) { return g(cfg, "network.width", d || 128); }
  function D(cfg) { return g(cfg, "network.depth", 2); }
  function dz(cfg) { var n = N(cfg); return n > 1 ? (5177.8 / n).toFixed(1) + " mm" : "5177.8 mm"; }
  function opt(cfg) { var o = g(cfg, "training.optimiser.name", "lbfgs_restarts"); return o.indexOf("broyden") >= 0 ? "SSBroyden" : "L-BFGS"; }

  function lossLine(cfg) {
    var n = g(cfg, "loss.name", "pooled");
    var map = {
      pooled: ["Pooled loss", "mean over n, j, d of (r/s_d)\u00b2"],
      cost_weighted: ["Cost-weighted loss", "lever arm \u00f7 track bend\nwindow " + JSON.stringify(g(cfg, "loss.momentum_window_gev", "")).replace(/[\[\]]/g, "").replace(",", " to ") + " GeV"],
      pooled_with_a_bias_term: ["Pooled + bias term", "+ \u03bb \u00b7 mean over bins of (mean r/s)\u00b2\nshare " + g(cfg, "loss.share_of_the_bias_term_at_round_one", "")],
      pooled_with_a_carried_defect_term: ["Pooled + carried defect", "+ \u03bb \u00b7 (\u03a3_steps carried defect / s)\u00b2\nshare " + g(cfg, "loss.share_of_the_carried_term_at_round_one", "")]
    };
    return map[n] || [n, ""];
  }

  function protocolSpec(cfg) {
    var p = g(cfg, "training.protocol.name", "rounds_on_own_predictions");
    var spr = g(cfg, "training.protocol.states_per_round", 32000);
    var txt = {
      rounds_on_own_predictions: ["Draw " + spr.toLocaleString("en") + " states", "round 1: reference states\nthen: own predictions"],
      rounds_on_one_fixed_draw: ["One fixed draw", "same tracks and planes every round\nafter round 1: " + (g(cfg, "training.protocol.states_after_round_one", "") === "the_reference_states_of_round_one" ? "reference states" : "own predictions")],
      rounds_on_own_predictions_with_residual_draws: ["Draw with residual weight", "half drawn by largest residual\nz-face steps weighted \u00d72"],
      rounds_on_whole_tracks_of_own_predictions: ["Draw whole tracks", Math.floor(spr / N(cfg)) + " tracks \u00d7 all " + N(cfg) + " planes\nround 1 reference, then own"],
      start_states_of_the_tracks: ["All start states", spr.toLocaleString("en") + " training tracks\nwith their labels"]
    }[p] || [p, ""];
    var rs = g(cfg, "training.restarts_per_round", 25), cap = g(cfg, "training.rounds_at_most", 80);
    return {
      nodes: [
        { id: "p", c: 0, r: 0, t: txt[0], sub: txt[1], k: "state", num: 1 },
        { id: "o", c: 1, r: 0, t: opt(cfg) + " restarts", sub: rs + " restarts \u00d7 200 iterations\nstates fixed in the round", k: "net", num: 2 },
        { id: "v", c: 2, r: 0, t: "Validate", sub: "chain all 1,463 val tracks\nendpoint median (\u00b5m)", k: "state", num: 3 },
        { id: "s", c: 3, r: 0, t: "Plateau rule", sub: "10-round window, 5 %\nheld 3 rounds \u00b7 cap " + cap, k: "use", num: 4 },
        { id: "t", c: 4, r: 0, t: "Test report", sub: "1,452 test tracks\nmedian, p95, by momentum", k: "state", num: 5 }
      ],
      edges: [["p", "o"], ["o", "v"], ["v", "s"], ["s", "t", "stopped"], ["s", "p", "not stopped: next round", "below"]]
    };
  }

  // ---------- the model types ----------
  var TYPES = {
    direct_summed: {
      title: "Direct states, summed end state",
      claim: "The network emits the stage states; the end state is summed from the field at those stages, so it never comes from the network directly.",
      step: function (cfg) {
        var qq = q(cfg), w = W(cfg), l = lossLine(cfg);
        return { nodes: [
          { id: "s0", c: 0, r: 0, t: "State S0 on z0", sub: "x, y, tx, ty, q/p\n+ plane z0", k: "state", num: 1 },
          { id: "in", c: 1, r: 0, t: "Scale inputs", sub: "\u00f7 spread of\nround-1 states", k: "phys", num: 2 },
          { id: "body", c: 2, r: 0, t: "Body " + D(cfg) + " \u00d7 " + w + " tanh", sub: "6 \u2192 " + w + " \u2192 " + w, k: "net", num: 3 },
          { id: "lin", c: 3, r: 0, t: "Linear layer", sub: w + " \u2192 " + (4 * qq) + " (4 per stage)", k: "net", num: 4 },
          { id: "of", c: 4, r: 0, t: "Direct states", sub: "Y_k = s_d \u00b7 raw\nfixed scale per component", k: "phys", num: 5 },
          { id: "y", c: 5, r: 0, t: "Stages Y1\u2026Y" + qq, sub: "states at z0 + c_k dz", k: "state", num: 6 },
          { id: "f", c: 5, r: 1, t: "Field lookups", sub: qq + " per step\nf(Y_k) = rates", k: "field", num: 7 },
          { id: "end", c: 4, r: 1, t: "Summed end state", sub: "S1 = S0 + dz \u03a3 b_k f(Y_k)\nGauss-Legendre weights", k: "phys", num: 8 },
          { id: "nx", c: 3, r: 1, t: "Next step: S1 \u2192 S0", sub: "repeat \u00d7 " + N(cfg) + ", dz = " + dz(cfg), k: "state", num: 9 },
          { id: "res", c: 5, r: 2, t: "Stage residual", sub: "r = Y \u2212 S0 \u2212 dz A f(Y)", k: "loss", num: 10 },
          { id: "loss", c: 4, r: 2, t: l[0], sub: l[1], k: "loss", num: 11 }
        ], edges: [["s0", "in"], ["in", "body"], ["body", "lin"], ["lin", "of", "raw"], ["of", "y"], ["y", "f"], ["f", "end", "rates"], ["end", "nx"], ["f", "res", "training", "train"], ["res", "loss", "", "train"]] };
      },
      steps: function (cfg) { return [
        "Take the state S0 and its plane z0.",
        "Scale the six inputs by the spread of the round-one states.",
        "Pass them through " + D(cfg) + " tanh layers of width " + W(cfg) + ".",
        "A linear layer gives 4 numbers per stage, " + 4 * q(cfg) + " in all.",
        "Multiply each by a fixed scale per component to get the stage states.",
        "The stages sit at z0 + c_k dz, the Gauss-Legendre nodes.",
        "Look up the field at each stage and compute the rates f(Y_k).",
        "Sum the end state with the Gauss-Legendre weights b_k.",
        "Repeat for " + N(cfg) + " steps. No gradient flows between steps.",
        "In training, compute the stage residual r = Y \u2212 S0 \u2212 dz A f(Y).",
        "The loss is the squared residual, scaled per component. No reference state enters it."
      ]; }
    },

    direct_predicted: {
      title: "Direct states, predicted end state",
      claim: "Here the network also emits the end state, so the chain carries the network's own end state instead of the summed one.",
      step: function (cfg) {
        var qq = q(cfg), w = W(cfg);
        return { nodes: [
          { id: "s0", c: 0, r: 0, t: "State S0 on z0", sub: "x, y, tx, ty, q/p\n+ plane z0", k: "state", num: 1 },
          { id: "body", c: 1, r: 0, t: "Body " + D(cfg) + " \u00d7 " + w + " tanh", sub: "inputs scaled first", k: "net", num: 2 },
          { id: "lin", c: 2, r: 0, t: "Linear layer", sub: w + " \u2192 " + (4 * qq + 4) + "\nstages + end state", k: "net", num: 3 },
          { id: "y", c: 3, r: 0, t: "Stages Y1\u2026Y" + qq, sub: "direct states", k: "state", num: 4 },
          { id: "end", c: 4, r: 0, t: "End state S1", sub: "emitted by the network", k: "net", num: 5 },
          { id: "nx", c: 5, r: 0, t: "Next step: S1 \u2192 S0", sub: "repeat \u00d7 " + N(cfg), k: "state", num: 6 },
          { id: "f", c: 3, r: 1, t: "Field lookups", sub: qq + " per step, training only", k: "field", num: 7 },
          { id: "res", c: 4, r: 1, t: "Residuals", sub: "stages: Y \u2212 S0 \u2212 dz A f\nend: S1 \u2212 S0 \u2212 dz \u03a3 b f", k: "loss", num: 8 },
          { id: "loss", c: 5, r: 1, t: "Pooled loss", sub: "stages and end state", k: "loss", num: 9 }
        ], edges: [["s0", "body"], ["body", "lin"], ["lin", "y"], ["y", "end"], ["end", "nx"], ["y", "f", "", "train"], ["f", "res", "", "train"], ["res", "loss", "", "train"]] };
      },
      steps: function () { return [
        "Take S0 on z0 and scale the inputs.",
        "The body and linear layer emit the stage states and the end state.",
        "The chain carries the emitted end state to the next step.",
        "In training, look up the field at the stages.",
        "Form the stage residuals and the end-state residual.",
        "The pooled loss squares both. The end state is never summed from the field."
      ]; }
    },

    straight_per_track: {
      title: "Straight line + correction, per-track scale",
      claim: "The network only learns the bend away from a straight line, scaled for each track by the field along that line.",
      step: function (cfg) {
        var qq = q(cfg), w = W(cfg), l = lossLine(cfg);
        return { nodes: [
          { id: "s0", c: 0, r: 0, t: "State S0 on z0", sub: "x, y, tx, ty, q/p\n+ plane z0", k: "state", num: 1 },
          { id: "body", c: 1, r: 0, t: "Body " + D(cfg) + " \u00d7 " + w + " tanh", sub: "inputs scaled first", k: "net", num: 2 },
          { id: "lin", c: 2, r: 0, t: "Linear layer", sub: w + " \u2192 " + (4 * qq) + "\nzero = straight line", k: "net", num: 3 },
          { id: "sl", c: 0, r: 1, t: "Straight line", sub: "S0 carried to each stage\nwith its slopes", k: "phys", num: 4 },
          { id: "sc", c: 1, r: 1, t: "Per-track scale", sub: "\u03ba \u00b7 q/p \u00b7 |B| along the line\n16 field samples", k: "field", num: 5 },
          { id: "of", c: 3, r: 0, t: "Add the correction", sub: "Y_k = line_k + scale \u00b7 raw_k", k: "phys", num: 6 },
          { id: "y", c: 4, r: 0, t: "Stages Y1\u2026Y" + qq, sub: "at z0 + c_k dz", k: "state", num: 7 },
          { id: "f", c: 5, r: 0, t: "Field lookups", sub: qq + " per step: f(Y_k)", k: "field", num: 8 },
          { id: "end", c: 5, r: 1, t: "Summed end state", sub: "S1 = S0 + dz \u03a3 b_k f(Y_k)", k: "phys", num: 9 },
          { id: "nx", c: 4, r: 1, t: "Next step: S1 \u2192 S0", sub: "repeat \u00d7 " + N(cfg) + ", dz = " + dz(cfg), k: "state", num: 10 },
          { id: "res", c: 5, r: 2, t: "Stage residual", sub: "r = Y \u2212 S0 \u2212 dz A f(Y)", k: "loss", num: 11 },
          { id: "loss", c: 4, r: 2, t: l[0], sub: l[1], k: "loss", num: 12 }
        ], edges: [["s0", "body"], ["body", "lin"], ["lin", "of", "raw"], ["s0", "sl"], ["sl", "sc"], ["sc", "of", "scale"], ["of", "y"], ["y", "f"], ["f", "end", "rates"], ["end", "nx", "", ""], ["end", "res", "training", "train"], ["res", "loss", "", "train"]] };
      },
      steps: function (cfg) { return [
        "Take S0 on z0 and scale the inputs.",
        "The body and linear layer give a raw correction per stage. A zero output means a straight line.",
        "Carry S0 along a straight line to each stage plane.",
        "Sample the field 16 times along that line to set this track's bend scale.",
        "Each stage is the straight line plus the scale times the raw output.",
        "Look up the field at the " + q(cfg) + " stages.",
        "Sum the end state with the Gauss-Legendre weights.",
        "Repeat for " + N(cfg) + " steps.",
        "Train on the squared stage residual" + (g(cfg, "loss.name") === "pooled_with_a_carried_defect_term" ? ", plus the carried-defect estimate of each track's endpoint error." : ".")
      ]; }
    },

    straight_global: {
      title: "Straight line + correction, global scale",
      claim: "Same as the per-track form, but one bend scale serves every track, so no field samples are needed to set it.",
      step: function (cfg) {
        var s = TYPES.straight_per_track.step(cfg);
        s.nodes.forEach(function (n) { if (n.id === "sc") { n.t = "Global scale"; n.sub = "median of the per-track scale\nover round-1 states"; n.k = "phys"; } });
        s.edges = s.edges.filter(function (e) { return !(e[0] === "sl" && e[1] === "sc"); });
        return s;
      },
      steps: function (cfg) { var s = TYPES.straight_per_track.steps(cfg); s[3] = "Use one fixed scale for all tracks: the median of the per-track scale over the round-one states."; return s; }
    },

    physics_layer: {
      title: "Physics layer inside the network",
      claim: "The last layer of the network applies the stage equations once. It has no weights, and the gradient flows through it into the body.",
      step: function (cfg) {
        var qq = q(cfg), w = W(cfg);
        return { nodes: [
          { id: "s0", c: 0, r: 0, t: "State S0 on z0", sub: "x, y, tx, ty, q/p\n+ plane z0", k: "state", num: 1 },
          { id: "body", c: 1, r: 0, t: "Body " + D(cfg) + " \u00d7 " + w + " tanh", sub: "+ linear \u2192 " + 4 * qq, k: "net", num: 2 },
          { id: "of", c: 2, r: 0, t: "Straight line + corr.", sub: "per-track scale\n16 field samples", k: "phys", num: 3 },
          { id: "y0", c: 3, r: 0, t: "Body stages Y\u2070", sub: "kept for analysis", k: "state", num: 4 },
          { id: "f0", c: 4, r: 0, t: "Field lookups", sub: qq + ": f(Y\u2070)", k: "field", num: 5 },
          { id: "map", c: 5, r: 0, t: "Stage map (layer)", sub: "Y\u00b9 = S0 + dz A f(Y\u2070)\nno weights", k: "phys", num: 6 },
          { id: "f1", c: 5, r: 1, t: "Field lookups", sub: qq + ": f(Y\u00b9)", k: "field", num: 7 },
          { id: "end", c: 4, r: 1, t: "Summed end state", sub: "S1 = S0 + dz \u03a3 b_k f(Y\u00b9_k)", k: "phys", num: 8 },
          { id: "nx", c: 3, r: 1, t: "Next step: S1 \u2192 S0", sub: "repeat \u00d7 " + N(cfg) + ", dz = " + dz(cfg), k: "state", num: 9 },
          { id: "res", c: 5, r: 2, t: "Residual after layer", sub: "r = Y\u00b9 \u2212 S0 \u2212 dz A f(Y\u00b9)\n= dz A (f(Y\u2070) \u2212 f(Y\u00b9))", k: "loss", num: 10 },
          { id: "loss", c: 4, r: 2, t: "Pooled loss", sub: "gradient flows back\nthrough the layer", k: "loss", num: 11 }
        ], edges: [["s0", "body"], ["body", "of", "raw"], ["of", "y0"], ["y0", "f0"], ["f0", "map", "rates"], ["map", "f1", "Y\u00b9"], ["f1", "end", "rates"], ["end", "nx"], ["f1", "res", "training", "train"], ["res", "loss", "", "train"]] };
      },
      steps: function (cfg) { return [
        "Take S0 on z0 and scale the inputs.",
        "The body emits a correction per stage.",
        "Add it to the straight line, scaled per track.",
        "These are the body's stages Y\u2070.",
        "Look up the field at Y\u2070 (" + q(cfg) + " lookups).",
        "Apply the stage equations once: Y\u00b9 = S0 + dz A f(Y\u2070). This layer has no weights.",
        "Look up the field at Y\u00b9 (" + q(cfg) + " more lookups, so 4 a step).",
        "Sum the end state from f(Y\u00b9).",
        "Repeat for " + N(cfg) + " steps.",
        "The residual after the layer is zero only when Y\u2070 is a fixed point of the stage map.",
        "The pooled loss trains the body through the layer."
      ]; }
    },

    learned_gain: {
      title: "Learned gain on a frozen chain network",
      claim: "A second, small network reads the stage residual of a frozen stage network and takes a correction off the summed end state.",
      step: function (cfg) {
        var qq = q(cfg), w = W(cfg, 64), rates = g(cfg, "network.inputs_of_the_gain", "") === "state_residual_and_rates";
        return { nodes: [
          { id: "s0", c: 0, r: 0, t: "State S0 on z0", sub: "x, y, tx, ty, q/p", k: "state", num: 1 },
          { id: "fr", c: 1, r: 0, t: "Frozen stage net", sub: "run " + g(cfg, "network.stage_network_of_the_run.key", "") + "\nnot trained", k: "use", num: 2 },
          { id: "y", c: 2, r: 0, t: "Stages Y", sub: "one call", k: "state", num: 3 },
          { id: "f", c: 3, r: 0, t: "Field lookups", sub: qq + ": F = f(Y)", k: "field", num: 4 },
          { id: "e", c: 4, r: 0, t: "Summed end E", sub: "E = S0 + dz \u03a3 b_k F_k", k: "phys", num: 5 },
          { id: "r", c: 3, r: 1, t: "Stage residual", sub: "r = Y \u2212 S0 \u2212 dz A F\nfree: F is known", k: "phys", num: 6 },
          { id: "gn", c: 2, r: 1, t: "Gain net " + D(cfg) + " \u00d7 " + w, sub: "in: S0, z0, r" + (rates ? ", F" : "") + "\nout: W (4 \u00d7 " + 4 * qq + ")", k: "net", num: 7 },
          { id: "d", c: 1, r: 1, t: "Correction", sub: "D = s \u00b7 W r", k: "phys", num: 8 },
          { id: "end", c: 5, r: 0, t: "End state", sub: "S1 = E \u2212 D", k: "state", num: 9 },
          { id: "k", c: 4, r: 2, t: "Analytic label K r", sub: "field Jacobians at Y\none linear solve", k: "field", num: 10 },
          { id: "loss", c: 2, r: 2, t: "Distilled gain loss", sub: "mean ((D \u2212 K r)/s)\u00b2\nlabel-free", k: "loss", num: 11 }
        ], edges: [["s0", "fr"], ["fr", "y"], ["y", "f"], ["f", "e"], ["e", "end"], ["f", "r"], ["r", "gn"], ["gn", "d"], ["d", "end", "subtract"], ["r", "k", "training", "train"], ["k", "loss", "", "train"], ["d", "loss", "", "train"]] };
      },
      steps: function () { return [
        "Take S0 on z0.",
        "Call the frozen stage network once.",
        "Its stages Y are fixed; only the gain network trains.",
        "Look up the field at the stages.",
        "Sum the end state E as usual.",
        "Form the stage residual r. It costs no extra lookups.",
        "The gain network reads S0, z0 and r and emits a matrix W.",
        "The correction is D = s \u00b7 W r.",
        "The end state is E \u2212 D.",
        "In training, the label K r is one linearised Newton step from the field Jacobians.",
        "The loss pulls D towards K r. No reference track enters it."
      ]; }
    },

    error_conditioned: {
      title: "Error-conditioned stage corrector",
      claim: "A corrector reads the residual of a frozen first call and moves the stages before the end state is summed, at the cost of a second set of lookups.",
      step: function (cfg) {
        var qq = q(cfg), w = W(cfg, 64);
        return { nodes: [
          { id: "s0", c: 0, r: 0, t: "State S0 on z0", sub: "x, y, tx, ty, q/p", k: "state", num: 1 },
          { id: "fr", c: 1, r: 0, t: "Frozen stage net", sub: "run " + g(cfg, "network.stage_network_of_the_run.key", "") + "\nfirst call", k: "use", num: 2 },
          { id: "y", c: 2, r: 0, t: "Stages Y", sub: "first call", k: "state", num: 3 },
          { id: "f", c: 3, r: 0, t: "Field lookups", sub: qq + ": f(Y)", k: "field", num: 4 },
          { id: "r", c: 4, r: 0, t: "Stage residual", sub: "r = Y \u2212 S0 \u2212 dz A f(Y)", k: "phys", num: 5 },
          { id: "cn", c: 5, r: 0, t: "Corrector " + D(cfg) + " \u00d7 " + w, sub: "in: S0, z0, r\nout: W (" + 4 * qq + " \u00d7 " + 4 * qq + ")", k: "net", num: 6 },
          { id: "y2", c: 5, r: 1, t: "Second stages", sub: "Y' = Y + s \u00b7 W r", k: "state", num: 7 },
          { id: "f2", c: 4, r: 1, t: "Field lookups", sub: qq + " more: f(Y')", k: "field", num: 8 },
          { id: "end", c: 3, r: 1, t: "Summed end state", sub: "S1 = S0 + dz \u03a3 b_k f(Y'_k)", k: "phys", num: 9 },
          { id: "nx", c: 2, r: 1, t: "Next step: S1 \u2192 S0", sub: "repeat \u00d7 " + N(cfg), k: "state", num: 10 },
          { id: "loss", c: 4, r: 2, t: "Pooled on both calls", sub: "residual of Y' and of Y\nmean of the two", k: "loss", num: 11 }
        ], edges: [["s0", "fr"], ["fr", "y"], ["y", "f"], ["f", "r"], ["r", "cn"], ["cn", "y2", "correction"], ["y2", "f2"], ["f2", "end"], ["end", "nx"], ["f2", "loss", "training", "train"]] };
      },
      steps: function () { return [
        "Take S0 on z0.",
        "Call the frozen stage network once.",
        "Keep its stages Y.",
        "Look up the field at Y.",
        "Form the stage residual r.",
        "The corrector reads S0, z0 and r and emits a square matrix W.",
        "Move the stages: Y' = Y + s \u00b7 W r.",
        "Look up the field again at Y'.",
        "Sum the end state from the corrected stages.",
        "Repeat along the chain.",
        "The loss is the pooled residual of both calls. The first call's half is a constant when it is frozen."
      ]; }
    },

    path_network: {
      title: "One-shot path network",
      claim: "One call emits every stage and plane state of the crossing. It learns from exact-scheme labels, and at use one Newton sweep of the stage equations polishes it.",
      step: function (cfg) {
        var w = W(cfg, 256), n = N(cfg), qq = q(cfg), unc = g(cfg, "loss.uncertainty", "none") === "diagonal";
        return { nodes: [
          { id: "s0", c: 0, r: 0, t: "First-plane state", sub: "x, y, tx, ty, q/p\none per track", k: "state", num: 1 },
          { id: "body", c: 1, r: 0, t: "Body " + D(cfg) + " \u00d7 " + w + " tanh", sub: "inputs scaled first", k: "net", num: 2 },
          { id: "lin", c: 2, r: 0, t: "Linear layer", sub: "\u2192 " + n * (qq + 1) * 4 + (unc ? " + \u03c3 outputs" : "") + "\n" + n + " \u00d7 (" + qq + " + 1) \u00d7 4", k: "net", num: 3 },
          { id: "of", c: 3, r: 0, t: "Straight line + corr.", sub: "one line through the region\n16 samples for the scale", k: "phys", num: 4 },
          { id: "p", c: 4, r: 0, t: "Whole path", sub: "all stages and planes\nno chain", k: "state", num: 5 },
          { id: "nw", c: 5, r: 0, t: "Newton sweep", sub: "at use, batched\nstage equations once", k: "use", num: 6 },
          { id: "out", c: 5, r: 1, t: "Last-plane state", sub: "after the sweep", k: "state", num: 7 },
          { id: "lab", c: 3, r: 1, t: "Label", sub: "exact Gauss-Legendre\nstage and plane states", k: "sup", num: 8 },
          { id: "loss", c: 4, r: 1, t: unc ? "Gaussian likelihood" : "Supervised path loss", sub: unc ? "((out \u2212 label)/\u03c3)\u00b2 + 2 log \u03c3" : "mean ((out \u2212 label)/s)\u00b2", k: "sup", num: 9 }
        ], edges: [["s0", "body"], ["body", "lin"], ["lin", "of", "raw"], ["of", "p"], ["p", "nw"], ["nw", "out"], ["p", "loss", "training", "train"], ["lab", "loss", "", "train"]] };
      },
      steps: function (cfg) { return [
        "Take the state on the first plane.",
        "Scale the inputs and pass them through the body.",
        "One linear layer emits every state of all " + N(cfg) + " steps at once.",
        "Each state is a straight line through the whole region plus a scaled correction.",
        "This is the whole path. It is not chained, and it contains no equation of motion.",
        "At use, one batched Newton sweep of the stage equations corrects it.",
        "The last plane gives the end state.",
        "In training, the labels are the exact scheme's states from the store.",
        "The loss is the squared difference to those labels."
      ]; }
    },

    whole_crossing: {
      title: "Whole-crossing network",
      claim: "The supervised twin: one call maps the first-plane state to the last-plane state, with no stages and no field inside.",
      step: function (cfg) {
        var w = W(cfg, 256);
        return { nodes: [
          { id: "s0", c: 0, r: 0, t: "First-plane state", sub: "x, y, tx, ty, q/p", k: "state", num: 1 },
          { id: "body", c: 1, r: 0, t: "Body " + D(cfg) + " \u00d7 " + w + " tanh", sub: "inputs scaled first", k: "net", num: 2 },
          { id: "lin", c: 2, r: 0, t: "Linear layer", sub: w + " \u2192 4", k: "net", num: 3 },
          { id: "out", c: 3, r: 0, t: "Last-plane state", sub: "direct states", k: "state", num: 4 },
          { id: "lab", c: 2, r: 1, t: "Label", sub: "sixth-order reference\nend state", k: "sup", num: 5 },
          { id: "loss", c: 3, r: 1, t: "Supervised endpoint", sub: "mean ((S \u2212 S_ref)/s)\u00b2", k: "sup", num: 6 }
        ], edges: [["s0", "body"], ["body", "lin"], ["lin", "out"], ["out", "loss", "training", "train"], ["lab", "loss", "", "train"]] };
      },
      steps: function () { return [
        "Take the state on the first plane.",
        "Pass it through the body.",
        "A linear layer gives four numbers.",
        "They are the state on the last plane.",
        "The label is the sixth-order reference end state.",
        "The loss is the squared difference, scaled per component."
      ]; }
    }
  };

  function typeOf(cfg) {
    var k = g(cfg, "network.kind", "stage_network"), of = g(cfg, "network.output_form", ""), sc = g(cfg, "network.scale_of_outputs", "");
    if (k === "stage_network_with_a_physics_layer") return "physics_layer";
    if (k === "stage_network_with_a_learned_gain") return "learned_gain";
    if (k === "error_conditioned_stage_network") return "error_conditioned";
    if (k === "path_network") return "path_network";
    if (k === "whole_crossing_network") return "whole_crossing";
    if (of === "direct_states") return g(cfg, "network.end_state") === "predicted" ? "direct_predicted" : "direct_summed";
    return sc === "per_track" ? "straight_per_track" : "straight_global";
  }

  var LEGEND = '<div class="ak-legend">' +
    '<span><i class="ak-sw ak-net"></i>has weights (trained)</span>' +
    '<span><i class="ak-sw ak-phys"></i>fixed arithmetic, no weights</span>' +
    '<span><i class="ak-sw ak-field"></i>field-map lookups</span>' +
    '<span><i class="ak-sw ak-loss"></i>label-free loss</span>' +
    '<span><i class="ak-sw ak-sup"></i>supervised loss and label</span>' +
    '<span><i class="ak-sw ak-use ak-dash"></i>frozen or at-use only</span>' +
    '<span><i class="ak-line"></i>dashed arrow: training only</span></div>';

  function render(type, cfg) {
    var T = TYPES[type]; cfg = cfg || {};
    var st = T.step(cfg), pr = protocolSpec(cfg);
    var ol = "<ol class=\"ak-steps\">" + T.steps(cfg).map(function (s) { return "<li>" + esc(s) + "</li>"; }).join("") + "</ol>";
    return '<div class="ak-wrap"><h3 class="ak-h">' + esc(T.title) + "</h3>" + LEGEND +
      '<figure class="ak-fig"><div class="ak-scroll">' + svg(st, T.claim) + "</div><figcaption>One step, in order. " + esc(T.claim) + "</figcaption></figure>" +
      '<details class="ak-det" open><summary>Order of operations</summary>' + ol + "</details>" +
      '<figure class="ak-fig"><div class="ak-scroll">' + svg(pr, "How one run trains, round by round") + "</div><figcaption>How the run trains, round by round.</figcaption></figure></div>";
  }

  // Minimal reader for the block YAML that configuration.yaml uses: nested maps by indent, "- x" lists, scalars.
  function parseYaml(text) {
    function scalar(v) {
      v = v.trim();
      if (v === "" || v === "null" || v === "~") return null;
      if (v === "true") return true; if (v === "false") return false;
      if (/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(v)) return Number(v);
      if (/^\[.*\]$/.test(v)) return v.slice(1, -1).split(",").filter(function (x) { return x.trim() !== ""; }).map(scalar);
      return v.replace(/^['"]|['"]$/g, "");
    }
    var root = {}, stack = [{ ind: -1, obj: root }], lines = String(text || "").split(/\r?\n/);
    for (var i = 0; i < lines.length; i++) {
      var raw = lines[i]; if (!raw.trim() || /^\s*#/.test(raw)) continue;
      var ind = raw.match(/^ */)[0].length, line = raw.trim();
      while (stack.length > 1 && ind <= stack[stack.length - 1].ind && !(line.indexOf("- ") === 0 && Array.isArray(stack[stack.length - 1].obj) && ind >= stack[stack.length - 1].ind)) stack.pop();
      var top = stack[stack.length - 1];
      if (line.indexOf("- ") === 0 && Array.isArray(top.obj)) { top.obj.push(scalar(line.slice(2))); continue; }
      var m = line.match(/^([^:]+):(.*)$/); if (!m) continue;
      var key = m[1].trim(), rest = m[2].trim();
      if (rest === "") {
        var next = lines.slice(i + 1).find(function (l) { return l.trim() && !/^\s*#/.test(l); }) || "";
        var child = next.trim().indexOf("- ") === 0 ? [] : {};
        top.obj[key] = child; stack.push({ ind: ind, obj: child });
      } else top.obj[key] = scalar(rest);
    }
    return root;
  }

  window.RKPINN_ARCH = { TYPES: TYPES, typeOf: typeOf, render: render, parseYaml: parseYaml, titles: Object.keys(TYPES).reduce(function (o, k) { o[k] = TYPES[k].title; return o; }, {}) };
})();
