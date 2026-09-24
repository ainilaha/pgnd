# PGND figure schemas

The two figures follow the current `manuscript/main.tex` formulation and the quadratic-potential implementation in `src/model.py`. All visible objects in the accompanying PowerPoint are editable. The history cells and interpolated input path are schematic, not quantitative data. Parameter subscripts are omitted in the diagram where they do not change the mathematical meaning.

## Figure 1: system and problem overview

---BEGIN PROMPT---

[Style & Meta-Instructions]

High-fidelity scientific schematic, technical vector illustration, clean white background, sharp boundaries, academic journal style. Strictly two-dimensional. Use a 1600 × 900 canvas and export at 3840 × 2160. Keep the rightmost 100 pixels empty. Use real, editable text. Do not add a large figure title, logos, watermarks, or visible zone names. Use the small panel letters a, b, and c. Do not add quantitative performance results.

[LAYOUT CONFIGURATION]

* **Selected Layout**: Linear Pipeline with a separate reference-label path and a bottom timing strip.
* **Composition Logic**: A physical system occupies the left third. A two-column observation-history matrix occupies the middle. An estimator and a single force prediction occupy the right half. The reference-force connection runs below the inference pipeline. A horizontal timeline lies beneath the history and prediction regions.
* **Color Palette**: Azure blue #347DA9 for observations and direct prediction, mint green #388E80 for latent dynamics and uplift, coral #CD725D for force targets and training-only links, slate #61737E for mechanics. Use pale versions of these colors for functional module fills.

[ZONE 1: LEFT - COUPLED PHYSICAL SYSTEM]

* **Container**: Open white region between x = 54 and 505, with no surrounding panel border.
* **Visual Structure**: Draw two vertical supports. Suspend one sagging messenger wire above a nearly horizontal contact wire. Connect the wires with seven thin vertical droppers. Place a short dark horizontal panhead directly under the contact wire. Draw a diamond linkage under the panhead, with four circular pin joints and a base resting on a double horizontal roof line. This is a topology schematic, not a scaled simulator mesh. Place a coral downward force arrow at the contact point and a blue response marker at the panhead. Place a horizontal train-motion arrow below the roof.
* **Key Text Labels**: "Coupled physical system", "Catenary (ANCF)", "F_c(t)", "a(t), u(t)", "Responses", "Train motion", "ANCF catenary + lumped pantograph", "Simulation outputs".

[ZONE 2: CENTER - AVAILABLE RESPONSE HISTORY]

* **Container**: Open region centered around x = 690.
* **Visual Structure**: Draw exactly sixteen horizontal rows and two narrow columns inside square matrix brackets. Fill the first column pale blue and the second pale green. Fill the final row with the darker blue and green. These cells represent sample availability, not numerical magnitudes. Place the first and last observation labels to the left of the array. Label the columns acceleration and uplift using a and u.
* **Key Text Labels**: "Response history", "x₀", "xₘ₋₁", "a", "u", "m historical samples", "Training-only scaling", "a, u only".

[ZONE 3: CENTER-RIGHT - PGND ESTIMATOR]

* **Container**: Thin blue rectangular boundary at approximately x = 917, y = 204, width = 293, height = 291.
* **Visual Structure**: Place the label PGND at the top. Inside, split the input into an upper blue direct-readout rectangle and a lower green latent-dynamics rectangle. Inside the green rectangle show z = [q, v]. Route both rectangles into one circular plus node at the right. Keep the upper and lower paths visibly separate until the plus node.
* **Key Text Labels**: "PGND", "Direct readout", "Latent dynamics", "z = [q, v]".

[ZONE 4: RIGHT AND LOWER CENTER - PREDICTION AND REFERENCE]

* **Container**: Open region to the right of the PGND estimator, with two small coral outlined reference/evaluation rectangles beneath it.
* **Visual Structure**: Show one large coral mathematical symbol for the next-sample force prediction. Place a reference-force rectangle below the estimator and a training/evaluation rectangle below the output. Use dashed coral connections exclusively for this lower reference/evaluation route. Do not feed reference force back into the observation matrix or inference modules.
* **Key Text Labels**: "F̂ₘ", "Next-sample contact force", "Newtons", "Reference label only", "Reference force Fₘ", "Training / evaluation".

[ZONE 5: BOTTOM - INFORMATION CUTOFF]

* **Container**: A horizontal timing strip from approximately x = 568 to 1416, y = 733, with labels and brackets beneath.
* **Visual Structure**: Place sixteen equally spaced blue observation dots followed by one coral target dot. Draw a vertical dashed coral boundary halfway between the final observation and the target. Bracket the observed interval separately from the one-sample forecast interval. Do not include a target-time response in the blue history.
* **Key Text Labels**: "Finite-history prediction", "t₀", "tₘ₋₁", "tₘ", "xₘ unavailable", "Observed span: (m − 1)Δt", "Δt", "m = 16, Δt = 1 ms: 15 ms observed + 1 ms forecast".

[CONNECTIONS]

1. A solid blue arrow points right from the panhead-response region into the observation-history matrix.
2. A solid blue arrow labeled "a, u only" points right from the matrix into PGND.
3. The PGND input splits into the upper direct branch and lower latent branch; both terminate at the circular plus node.
4. A solid slate arrow points right from the plus node to the single force-prediction symbol.
5. A dashed coral polyline leaves the physical contact point, bends down beside the physical-system region, then points right into the reference-force rectangle.
6. A dashed coral arrow links reference force to training/evaluation; a second dashed coral arrow descends from the prediction into that rectangle.
7. No physical feedback loop or persistent state across windows is drawn.

---END PROMPT---

**Suggested caption.** Pantograph–catenary system and finite-history contact-force estimation. Simulation-derived panhead acceleration and uplift form an m-sample input window. PGND combines direct observation-conditioned prediction with a latent dynamical correction to estimate the following force sample. Reference force is used only for training or evaluation. The target-time response is excluded. For m = 16 and Δt = 1 ms, the observed history spans 15 ms and the forecast horizon is 1 ms. The physical drawing and history array are schematic.

## Figure 2: method

---BEGIN PROMPT---

[Style & Meta-Instructions]

Use the same white background, type hierarchy, line weights, and blue/green/coral/slate palette as Figure 1. Strictly two-dimensional, sharp vector boundaries, editable labels and equations. Use a 1600 × 900 canvas and export at 3840 × 2160. Keep the rightmost 100 pixels empty. Do not add a large figure title, logo, watermark, visible zone names, or performance claims. Use the small panel letters a, b, and c.

[LAYOUT CONFIGURATION]

* **Selected Layout**: Parallel/Dual-Stream with a central linear ODE trajectory and a bottom training objective.
* **Composition Logic**: The left side holds standardized history, a shared encoder, and an initializer. A long upper blue route carries only the latest available encoding to the direct readout. The middle blue route constructs a continuous input for a large green ODE region. The ODE output enters a separate green state decoder. The two scalar readout outputs meet at one plus node on the right, followed by inverse scaling. An optional initializer extension sits at lower left; the objective sits below the central ODE.
* **Color Palette**: Azure blue #347DA9 for encoded observations and direct prediction; mint green #388E80 for latent evolution and decoding; coral #CD725D for training dependencies and the final force output; slate #61737E for initialization and annotations.

[ZONE 1: LEFT - ENCODING AND INITIALIZATION]

* **Container**: Open left region spanning approximately x = 51 to 477.
* **Visual Structure**: Draw a small two-column, sixteen-row history array with square brackets. To its right, place a blue shared-encoder rectangle. Below that rectangle place a slate initializer rectangle. Draw a branch from the available input to the initializer and mark it as the first standardized observation only. The encoder must be shared by both readout paths.
* **Key Text Labels**: "Encoding and initialization", "Standardized response history", "x̃₀ … x̃ₘ₋₁", "Shared encoder E", "hᵢ = E(x̃ᵢ)", "Initializer E₀", "z₀ = E₀(x̃₀)".

[ZONE 2: CENTER - ENCODED INPUT AND STRUCTURED EVOLUTION]

* **Container**: A thin blue rectangle above a larger pale-green rectangle, centered between x = 550 and 1098.
* **Visual Structure**: In the upper rectangle draw a small piecewise-linear blue path through discrete dots, followed by a horizontal final segment. Mark the start of the horizontal segment with a dashed coral vertical line. In the green rectangle draw a horizontal sequence of circular latent-state nodes z₀, zᵢ, an ellipsis, and zₘ. Connect them with rightward green arrows. Below the nodes place the two exact vector-field equations. Color and label the damping, restoring, driving, and residual terms separately. Show the positive-semidefinite constraints and quadratic potential below the equation.
* **Key Text Labels**: "Structured latent evolution", "Encoded input h(τ)", "Interpolate history, then hold hₘ₋₁", "ODE integration (RK4)", "q′ = v", "v′ = −Dv − Kq + Bh(τ) + r", "Damping", "Restoring", "Driving", "Residual", "D ⪰ 0", "K ⪰ 0", "V(q) = ½ qᵀKq", "r = MLP(q, v, h, τ)".

[ZONE 3: RIGHT - ADDITIVE READOUT]

* **Container**: Two vertically separated rectangles above one circular plus node and one inverse-scaling rectangle, within x = 1202 to 1481.
* **Visual Structure**: Place a blue direct-prediction rectangle at the upper right and a green latent-correction rectangle beneath it. Route the blue rectangle around the far right into the plus node. Route the green rectangle straight downward into the plus node. From the plus node draw a downward arrow into a coral inverse-scaling rectangle. The plus node combines two scalar standardized-force contributions.
* **Key Text Labels**: "Additive readout", "Latest available encoding hₘ₋₁", "Direct prediction", "G(hₘ₋₁)", "Latent correction", "H(zₘ)", "+", "Inverse scaling", "Force F̂ₘ (N)".

[ZONE 4: LOWER LEFT - OPTIONAL WARM-UP]

* **Container**: A dashed slate rectangular outline, physically separate from the main initializer.
* **Visual Structure**: Arrange a short input-prefix label, a small E₀ block, and a latent-state symbol in one horizontal row. Use two short slate arrows between them. This inset describes an alternative to the main single-observation initializer, not an additional active input.
* **Key Text Labels**: "Optional warm-up (untested)", "K₀ + 1 inputs", "E₀", "z", "Begin at the last warm-up sample".

[ZONE 5: BOTTOM CENTER - LEARNING OBJECTIVE]

* **Container**: A coral dashed rectangle below the ODE region.
* **Visual Structure**: Place one concise loss equation in the rectangle and the definition of the residual penalty below it. A standardized-reference-force label sits to the left. A dashed coral path descends from the readout sum, bends left, and enters the loss rectangle from the right. A short dashed coral arrow enters from the residual definition above. Put a small line-style legend at lower left.
* **Key Text Labels**: "Loss = standardized force MSE + λ R", "R = mean ‖r‖² over windows and evolved times", "Standardized reference force F̃ₘ", "Final-target supervision with autograd + Adam", "Inference", "Training".

[CONNECTIONS]

1. A solid blue arrow connects standardized history to the shared encoder.
2. A slate branch labeled x̃₀ connects only the first available observation to E₀; a slate arrow carries the initialized state into the first ODE node.
3. The encoder output splits at a blue dot. One route rises and runs across the top to G(hₘ₋₁). The other route enters the interpolation/hold rectangle.
4. A solid blue downward arrow connects the encoded input path to the ODE region.
5. Rightward green arrows connect latent-state nodes. A green arrow connects zₘ to H(zₘ).
6. G and H enter the same plus node through separate blue and green routes. A slate arrow carries their sum into inverse scaling.
7. Dashed coral arrows carry the final standardized prediction, standardized reference target, and residual penalty into the objective. No reference force or target-time observation connects to the encoder.
8. The warm-up inset remains separate from the active K₀ = 0 path. If it is adopted, integration starts at τ_K₀ and its prefix belongs to the same m-observation budget.

---END PROMPT---

**Suggested caption.** Observation-conditioned PGND with a shared observation encoder, a quadratic-potential latent ODE, and an additive force readout. Encoded observations are interpolated over the available history and held at the final encoding over the prediction interval. The main path initializes from the first observation and integrates q′ = v and v′ = −Dv − Kq + Bh(τ) + r with RK4. The direct predictor and latent decoder sum in standardized-force space before inverse scaling. Training supervises the final force target and regularizes the squared residual norm. The warm-up initializer is a separate untested extension. Primes denote derivatives with respect to solver time; the latent coordinates and operators are not identified physical states or mechanical parameters.

### Scientific notation and scope

- The factorization is D = L_D L_Dᵀ and K = L_K L_Kᵀ; both matrices are positive semidefinite.
- The latent and encoded dimensions in the reference implementation are d = d_e = 16. The residual depends on q, v, h, and solver time τ.
- The exact residual penalty averages over windows and evolved sample times, sums over latent components, and excludes the initial state. It is not a physical force-balance loss.
- The additive readout is implemented in `src/model.py` as an option. The manuscript still treats its quantitative evaluation as pending; the figures make no accuracy or robustness claim. Warm-up remains unimplemented.
- Per-slide speaker notes include source equation labels and further scope details.
