/* Benchmark data, transcribed from the InWorld technical report (Tables 1–3, Figures 4 and 8).
   Edit here; every chart and table on the page is generated from these arrays. */
window.INWORLD_DATA = {
  wbench: {
    dims: [
      { key: "overall", label: "Overall" },
      { key: "quality", label: "Quality" },
      { key: "setting", label: "Setting" },
      { key: "interaction", label: "Interaction" },
      { key: "consistency", label: "Consistency" },
      { key: "physical", label: "Physical" }
    ],
    // model, Overall, Quality, Setting, Interaction, Consistency, Physical
    rows: [
      ["Murea-Pro", 81.13, 79.97, 86.48, 82.60, 82.16, 74.42, true],
      ["Kling 3.0", 79.42, 80.02, 90.96, 73.08, 83.86, 69.19],
      ["Wan 2.7", 78.37, 80.99, 91.45, 72.09, 75.77, 71.57],
      ["H3 FL2VA", 77.24, 79.15, 86.22, 71.05, 83.09, 66.68],
      ["Seedance 1.5", 76.27, 81.94, 82.94, 68.34, 79.91, 68.24],
      ["LTX 2.5", 74.80, 79.11, 87.91, 56.93, 80.81, 69.26],
      ["HY-Video 1.5", 74.30, 76.61, 85.56, 54.70, 87.48, 67.15],
      ["Cosmos3-Super", 72.44, 74.68, 89.40, 44.90, 84.30, 68.90],
      ["LTX 2.3", 70.96, 77.05, 85.23, 49.36, 78.05, 65.13],
      ["Cosmos 2.5", 70.35, 71.67, 83.34, 43.47, 86.28, 67.01],
      ["LongCat-Video", 69.91, 77.17, 72.32, 45.05, 86.62, 68.42],
      ["Cosmos3-Nano", 69.28, 74.98, 86.60, 36.50, 84.45, 63.85],
      ["YUME 1.5", 68.93, 77.60, 72.38, 48.38, 80.85, 65.42],
      ["Kairos 3.0", 65.73, 73.09, 70.33, 41.64, 83.16, 60.46]
    ],
    heatModels: ["Murea-Pro", "Kling 3.0", "Wan 2.7", "H3 FL2VA", "LTX 2.5"],
    heat: [
      ["Quality", [
        ["Aesthetic quality", 60.83, 61.32, 59.58, 61.32, 62.09],
        ["Imaging quality", 67.67, 67.66, 68.06, 67.77, 61.20],
        ["Temporal flickering", 93.77, 94.50, 93.04, 94.52, 96.41],
        ["Dynamic degree", 89.62, 89.93, 99.31, 82.35, 88.24],
        ["Motion smoothness", 97.98, 97.90, 96.55, 98.06, 98.95],
        ["HPSv3 quality", 69.96, 68.81, 69.43, 70.85, 67.79]
      ]],
      ["Setting", [
        ["Scene adherence", 83.86, 89.04, 88.32, 82.53, 82.66],
        ["Subject adherence", 89.10, 92.88, 94.58, 89.91, 93.15]
      ]],
      ["Interaction", [
        ["Navigation trajectory", 71.79, 70.32, 65.97, 69.56, 62.31],
        ["Event edit adherence", 90.92, 81.40, 84.00, 85.52, 65.02],
        ["Subject action adherence", 87.04, 85.60, 83.40, 79.11, 68.14],
        ["Perspective switch adherence", 80.65, 55.00, 55.00, 50.00, 32.26]
      ]],
      ["Consistency", [
        ["Background consistency", 90.71, 92.71, 89.48, 90.51, 89.04],
        ["Segment continuity", 96.54, 92.71, 65.62, 97.23, 86.85],
        ["Perspective consistency", 68.46, 76.75, 62.18, 72.58, 66.21],
        ["Subject consistency", 88.96, 88.45, 88.73, 89.98, 86.98],
        ["Geometric consistency", 87.06, 89.44, 82.63, 88.11, 90.62],
        ["Photometric consistency", 77.02, 80.40, 75.49, 77.94, 84.38],
        ["Spatial consistency", 74.25, 75.25, 71.00, 74.33, 71.40],
        ["Gated spatial consistency", 74.25, 75.14, 71.00, 74.04, 71.01]
      ]],
      ["Physical", [
        ["Visual plausibility", 60.19, 60.37, 59.84, 60.01, 59.31],
        ["Causal fidelity", 88.65, 78.00, 83.30, 73.34, 79.20]
      ]]
    ]
  },

  opens2v: {
    // key, label, decimals, higher is better
    metrics: [
      ["total", "Total", 2, true],
      ["aes", "Aesthetics", 2, true],
      ["smooth", "Motion smoothness", 2, true],
      ["motion", "Motion amplitude", 2, true],
      ["face", "FaceSim", 2, true],
      ["gme", "GmeScore", 2, true],
      ["nexus", "NexusScore", 2, true],
      ["natural", "NaturalScore", 2, true]
    ],
    // model, group (M = ours, C = commercial, O = open-source), clips N of 180, Total, Aes, Smooth, Motion, FaceSim, Gme, Nexus, Natural
    rows: [
      ["Murea-Fast (8 steps)", "M", 180, 64.98, 52.20, 97.91, 19.78, 58.21, 63.69, 51.66, 86.44],
      ["Seedance 2.0", "C", 148, 67.13, 49.90, 98.25, 22.24, 60.85, 68.80, 51.42, 92.06],
      ["HappyHorse 1.1", "C", 177, 63.04, 52.55, 97.84, 23.27, 56.12, 68.97, 43.25, 83.95],
      ["Vidu Q3 Mix", "C", 174, 61.99, 55.84, 98.37, 24.05, 39.62, 68.76, 47.95, 87.12],
      ["Kling V3 Omni", "C", 167, 61.33, 48.24, 92.78, 21.03, 52.88, 67.24, 47.06, 81.54],
      ["HappyHorse 1.0", "C", 163, 61.16, 52.71, 97.66, 27.40, 36.75, 68.07, 46.64, 89.47],
      ["Kling O3", "C", 168, 60.70, 48.88, 94.08, 20.41, 53.00, 67.39, 45.45, 79.37],
      ["Wan 2.7 R2V", "C", 159, 60.25, 47.02, 95.84, 20.53, 53.80, 63.75, 44.63, 80.14],
      ["Vidu Q3 Turbo", "C", 170, 58.94, 52.38, 98.19, 34.67, 58.09, 69.07, 41.22, 65.93],
      ["Kling 1.6", "C", 166, 58.54, 44.49, 86.88, 40.75, 41.37, 66.42, 44.72, 84.19],
      ["Grok Video", "C", 180, 53.42, 54.58, 99.46, 21.80, 4.66, 72.79, 34.30, 90.65],
      ["Pika 2.1", "C", 176, 53.33, 47.00, 87.18, 26.60, 28.46, 69.59, 45.15, 70.74],
      ["Vidu 2.0", "C", 180, 52.95, 41.40, 90.22, 13.68, 36.68, 68.49, 43.08, 68.61],
      ["Bernini", "O", 180, 63.21, 46.89, 90.11, 25.45, 58.93, 66.91, 48.96, 84.12],
      ["MiniMax-H3", "O", 180, 61.01, 48.13, 99.03, 29.63, 40.44, 69.19, 41.05, 92.41],
      ["VACE-14B", "O", 180, 59.28, 47.21, 94.92, 15.02, 55.09, 67.28, 44.08, 74.26],
      ["Phantom-14B", "O", 180, 58.19, 45.96, 96.17, 33.08, 53.59, 70.66, 37.44, 73.84],
      ["LTX-2.5 MSR LoRA (30 steps)", "O", 180, 57.71, 49.26, 97.67, 24.32, 36.32, 68.10, 46.36, 78.24],
      ["RefAlign-14B", "O", 180, 56.60, 45.57, 97.52, 15.33, 58.14, 59.66, 42.13, 66.39],
      ["Phantom-1.3B", "O", 180, 55.72, 46.27, 93.29, 14.01, 50.02, 69.65, 41.46, 65.79],
      ["MAGREF-480P", "O", 180, 55.04, 45.12, 92.75, 21.92, 35.55, 70.55, 43.48, 73.10],
      ["SkyReels-A2-P14B", "O", 180, 52.91, 37.92, 86.63, 26.48, 46.31, 65.17, 43.76, 63.66],
      ["VACE-1.3B", "O", 180, 51.05, 48.23, 97.19, 18.83, 20.58, 71.26, 37.93, 70.28],
      ["VACE-P1.3B", "O", 180, 50.79, 47.28, 96.87, 12.01, 17.89, 71.41, 41.06, 70.05]
    ]
  },

  os2vpro: {
    // key, label, decimals, higher is better, ranked
    metrics: [
      ["pro", "Pro aggregate", 1, true, true],
      ["opens2v", "OpenS2V Total", 2, true, true],
      ["fid", "Per-reference fidelity (Fid)", 1, true, true],
      ["intid", "Intra-clip identity (IntID)", 1, true, true],
      ["decay", "Three- to one-reference fidelity (Decay)", 2, true, true],
      ["vtf", "Task following (VTF)", 1, true, true],
      ["bind", "Attribute binding (Bind)", 1, true, true],
      ["anat", "Anatomical consistency (Anat)", 1, true, true],
      ["limb", "Forearm-length variation (Limb)", 2, false, true],
      ["vq", "Technical quality (VQ)", 1, true, true],
      ["iqa", "Single-frame image quality (IQA)", 1, true, true],
      ["sharp", "Relative sharpness change (Sharp)", 2, true, true],
      ["flick", "Temporal flicker (Flick)", 2, false, true],
      ["usable", "Usable clips (Usable)", 1, true, true],
      ["atq", "Audio production quality (ATQ)", 1, true, true],
      ["vesa", "Event–sound alignment (VESA)", 1, true, true],
      ["wall", "Seconds per clip (Wall)", 1, false, false]
    ],
    rows: [
      ["Murea-Fast (8 steps)", "M", 69.9, 64.98, 46.0, 89.0, 0.58, 95.5, 19.6, 76.2, 0.34, 21.6, 39.9, -0.28, 0.06, 91.1, 59.7, 75.0, 43.7],
      ["Kling V3 Omni", "C", 70.1, 61.33, 41.9, 84.7, 0.56, 77.1, 22.6, 82.1, 0.31, 19.1, 41.9, -0.04, 0.06, 88.0, 66.3, 88.7, 131.8],
      ["Vidu Q3 Turbo", "C", 69.7, 58.94, 39.8, 89.0, 0.62, 74.6, 21.8, 77.6, 0.30, 20.0, 48.7, -0.11, 0.07, 93.5, 59.2, 83.3, 107],
      ["Kling O3", "C", 69.4, 60.70, 41.1, 83.8, 0.56, 77.3, 20.9, 77.0, 0.32, 19.0, 42.4, -0.05, 0.06, 88.7, 65.7, 81.1, null],
      ["HappyHorse 1.1", "C", 67.7, 63.04, 41.2, 84.8, 0.62, 74.7, 18.7, 73.9, 0.33, 21.5, 46.4, -0.38, 0.07, 94.3, 62.6, 87.2, 126.8],
      ["Vidu Q3 Mix", "C", 67.4, 61.99, 41.8, 81.7, 0.64, 76.8, 15.2, 68.4, 0.36, 23.2, 46.6, 0.04, 0.08, 96.0, 66.0, 76.7, 135],
      ["Grok Video", "C", 67.0, 53.42, 30.1, 82.4, 0.75, 70.0, 20.2, 79.2, 0.30, 23.0, 51.4, -0.08, 0.05, 91.7, 61.0, 90.5, 64.3],
      ["Wan 2.7 R2V", "C", 65.6, 60.25, 39.9, 78.5, 0.77, 72.2, 17.8, 74.6, 0.38, 21.3, 51.3, -0.12, 0.10, 96.2, 63.6, 88.7, 106.6],
      ["HappyHorse 1.0", "C", 64.6, 61.16, 39.3, 78.4, 0.68, 78.8, 16.0, 67.7, 0.33, 21.1, 47.8, -0.51, 0.08, 94.5, 76.6, 89.2, 106.4],
      ["Vidu 2.0", "C", 44.1, 52.95, 36.0, 79.2, 0.63, 63.0, 20.9, 61.4, 0.30, 15.8, 38.1, -0.04, 0.04, 88.3, null, null, 55],
      ["Pika 2.1", "C", 41.6, 53.33, 37.6, 74.6, 0.73, 68.6, 19.5, 49.0, 0.33, 15.8, 33.1, -0.14, 0.07, 90.9, null, null, null],
      ["Kling 1.6", "C", 38.7, 58.54, 38.8, 73.7, 0.61, 74.8, 18.9, 51.4, 0.40, 16.9, 33.0, -0.11, 0.13, 93.4, null, null, null],
      ["MiniMax-H3", "O", 64.9, 61.01, 32.1, 83.5, 0.72, 81.0, 16.0, 77.0, 0.37, 20.0, 37.0, -0.29, 0.08, 86.1, 64.7, 86.1, 494.2],
      ["LTX-2.5 MSR LoRA", "O", 62.2, 57.71, 37.9, 84.0, 0.56, 71.5, 17.4, 58.0, 0.34, 20.7, 38.3, -0.46, 0.09, 95.0, 66.8, 74.2, 197.4],
      ["VACE-14B", "O", 45.6, 59.28, 42.1, 85.8, 0.57, 65.5, 19.4, 64.1, 0.33, 19.7, 46.5, -0.04, 0.05, 91.7, null, null, 357.4],
      ["RefAlign-14B", "O", 45.5, 56.60, 39.2, 87.7, 0.67, 62.7, 19.6, 54.7, 0.33, 19.6, 45.7, -0.03, 0.05, 90.6, null, null, 473.1],
      ["Phantom-1.3B", "O", 44.3, 55.72, 37.3, 81.8, 0.77, 62.8, 19.7, 59.4, 0.37, 19.0, 50.7, 0.09, 0.05, 83.9, null, null, 102.9],
      ["Bernini", "O", 43.8, 63.21, 44.8, 81.7, 0.58, 71.8, 17.2, 62.6, 0.35, 19.1, 44.7, -0.19, 0.07, 92.2, null, null, 527.5],
      ["MAGREF-480P", "O", 43.6, 55.04, 35.1, 76.6, 0.70, 69.1, 19.2, 49.4, 0.32, 17.6, 44.6, 0.04, 0.07, 89.4, null, null, 206.7],
      ["Phantom-14B", "O", 43.5, 58.19, 34.6, 84.4, 0.69, 67.9, 18.3, 51.8, 0.34, 17.7, 49.1, -0.14, 0.08, 89.4, null, null, 421.3],
      ["VACE-P1.3B", "O", 42.9, 50.79, 33.1, 80.1, 0.63, 61.7, 18.9, 60.7, 0.30, 18.8, 48.4, 0.03, 0.04, 81.7, null, null, 112.0],
      ["VACE-1.3B", "O", 42.5, 51.05, 33.4, 80.4, 0.61, 62.4, 17.2, 58.6, 0.31, 18.9, 50.1, 0.00, 0.06, 88.3, null, null, 112.8],
      ["SkyReels-A2-P14B", "O", 41.9, 52.91, 40.6, 77.6, 0.55, 63.8, 18.4, 48.2, 0.37, 18.3, 42.6, 0.13, 0.06, 88.3, null, null, 265.4]
    ]
  },

  latency: {
    baseline: { label: "Baseline", note: "50 NFE, FlashAttention-3", value: 231.6 },
    steps: [
      { label: "Few-step distillation", note: "50 to 4 NFE", value: 23.2, step: "10.0×" },
      { label: "+ SageAttention", note: "Efficient attention", value: 17.9, step: "1.30×" },
      { label: "+ Adaptive BlockCache", note: "Cache acceleration", value: 14.7, step: "1.22×" },
      { label: "+ 1.5× super-resolution", note: "Generate at lower resolution", value: 6.96, step: "2.11×" }
    ]
  }
};
