# Proofreading report

## Market Equilibrium Effects of Storage Staircase Bidding: More Than Revenue at Stake

Reviewed from the 12-page PDF dated July 25, 2026.

## Overall assessment

The paper has a clear motivation, a sensible high-level organization, and readable figures. It is not yet ready for submission without revision. Most language problems are straightforward, but several methodological claims need clarification or qualification. The most important issue is that the sequential MARL formulation is not demonstrated to be equivalent to the full-horizon day-ahead EPEC introduced in Section II.

## High-priority logical and methodological issues

### 1. The sequential MARL environment may not solve the stated day-ahead EPEC

Section II defines a full-horizon problem: every ESS submits bids over all \(t\in T\), and the DSO clears an OPF over \(T\) with intertemporal SoC constraints. Section III-A instead models the interaction as 24 sequential steps. Algorithm 1 says that at each step an agent submits the current bid, after which the environment solves (11)-(29), returns the current reward, and updates SoC.

It is unclear how the lower-level, full-horizon optimization is solved before future bids have been produced. If clearing is hourly, the problem is no longer the full-horizon day-ahead OPF in (11)-(29). If all 24 bids are collected before one clearing, the algorithm and immediate-reward description are inaccurate. Please state the exact timing and information structure, then either prove equivalence or describe the MARL environment as an approximation.

This also affects the interpretation of "day-ahead": a genuinely day-ahead market normally receives the full 24-hour bid profile before clearing, rather than revealing hourly outcomes between successive bid decisions.

### 2. MATD3 training does not by itself establish a Nash equilibrium

Equations (40)-(41) define a policy-space equilibrium, but standard MATD3 training provides no guarantee that the learned policies satisfy those best-response conditions. The manuscript repeatedly calls the learned outcome an equilibrium without reporting an equilibrium-gap, exploitability, or unilateral-deviation test.

Suggested remedy: call the MATD3 result a "learned policy profile" or "approximate equilibrium candidate" and validate it by holding the other agents fixed and optimizing each agent's deviation. Report the maximum or average unilateral profit gain. Also distinguish:

- the scenario-specific bid-space Nash equilibrium in (31); and
- the ex-ante policy game over a scenario distribution in (40)-(41).

These are not automatically the same equilibrium concept.

### 3. "Market efficiency" is not established by the reported system-cost metric

The DSO objective in (11) includes strategic bid prices. Bid payments are transfers and need not equal physical production costs or social costs. Comparing the value of this bid-dependent objective across bidding formats therefore does not, by itself, establish greater market efficiency.

Please define exactly what is plotted as "system cost." For an efficiency claim, consider reporting a bid-independent metric evaluated at the cleared dispatch, such as true generation cost plus degradation cost, total social welfare, production-cost uplift relative to a centralized benchmark, or deadweight loss.

### 4. Results from nonconverged GS-FPI iterates are not equilibrium outcomes

The final iterate of a GS-FPI run that fails (44) is not a valid fixed-point equilibrium. Revenue, cost, and network-binding results from such iterates can illustrate algorithmic failure, but they should not be used as evidence about equilibrium market outcomes or the economic superiority of one bidding format.

Relabel these results as diagnostics of the solver's terminal iterates, and avoid statements such as "inefficient market clearing caused by unstable bidding interactions." Each lower-level clearing may still be optimal for the submitted bids; what failed is the outer equilibrium iteration.

### 5. Converged-case comparisons may suffer from selection bias

The paper reports 4 converged test cases for single-point bidding and 10 for staircase bidding. If each format converges on a different subset of scenarios, comparing their converged-case averages is not a paired economic comparison. The same concern applies to GS-FPI solution times averaged only over converged runs.

Report:

- results on the common subset of scenarios where both formats converge;
- paired per-scenario differences;
- all-scenario algorithmic success rates separately; and
- means, dispersion, and confidence intervals over random seeds where applicable.

### 6. The single-point benchmark is underspecified and may not be comparable

Section IV-A5 says only that the staircase constraints are replaced by bounds on a quantity-price pair. The exact upper- and lower-level equations are needed. In particular, the staircase model is forced to span the ESS's full rated charging and discharging ranges, explicitly preventing capacity withholding, while the single-point model appears to choose one quantity. The reported differences may therefore combine bidding granularity with different capacity-withholding rights.

State how a signed single quantity represents both charging and discharging, how step acceptance works, how SoC feasibility is enforced, and whether both formats expose the same physical capacity to the market.

### 7. The perturbation experiment is not yet an apples-to-apples robustness test

For single-point bidding, both coordinates of the entire bid are perturbed. For staircase bidding, only one breakpoint on each branch is perturbed. A percentage of each format's feasible range is not necessarily an equal perturbation in function space, especially when the action dimensions differ.

Define the feasibility-preserving procedure precisely, including how monotonicity and the zero-junction constraint remain satisfied. Consider matching perturbations by an \(L_1\) or \(L_2\) distance between bid curves, displaced energy, or induced offer-cost change. Report the number of random draws, seeds, and uncertainty intervals.

### 8. Several empirical interpretations are stronger than the experiment supports

On page 8, wide early-hour bids are attributed to "high uncertainty about how the day will unfold," and single-point bids are said to be unable to adapt to "unexpected system conditions." Yet the study explicitly excludes forecast errors and uses fixed load and renewable profiles. These interpretations should be removed or reframed as hypotheses.

Similarly, the conclusion says the strategy is robust to "market changes," but the robustness experiment perturbs a participant's bid, not market conditions. Use "competitor-bid perturbations" or expand the experiment.

### 9. Important reproducibility details are missing

Please add:

- the optimization solver used for the KKT-embedded best-response problems;
- whether each nonconvex best response is solved globally or locally, with tolerances/gaps;
- neural-network architectures, replay-buffer size, batch size for fine-tuning, exploration noise, target-update frequency, and random seeds;
- the number of independent MATD3 runs and the definition of the Fig. 10 and Fig. 11 bands/whiskers;
- the definition of DLMP standard deviation (over buses, hours, scenarios, or some combination);
- scenario-selection dates/procedure and train-test separation; and
- hardware/software details for Table VI.

The KKT conditions establish lower-level LP optimality, but they do not make the upper-level best-response problem convex. If only local solutions are obtained, the Nash-equilibrium wording must be qualified.

### 10. End-of-horizon observations are undefined

Equation (33) uses \(p_{t:t+k}\) with \(k=2\). At \(t=22\) and \(t=23\), the terms for hours 24 and 25 are outside \(T=\{0,\ldots,23\}\). State whether the profiles wrap to the next day, are padded, truncated, or supplied from the following day.

### 11. Scalability claims should be narrowed

One additional 118-bus case demonstrates applicability to a larger instance but does not establish a scaling law. Moreover, the statement that network variables scale with \(|N||T|\) omits line-flow variables, which scale with \(|E||T|\). Recast this as a larger-case computational comparison and give the complete dimensional dependence.

## Definite language, grammar, and presentation corrections

### Page 1

- Abstract: "As the increasing penetration of intermittent renewable energy drives..." -> "As increasing penetration of intermittent renewable energy increases the need for grid flexibility..."
- "at the distribution-level" -> "at the distribution level" (use the hyphen only before a noun, as in "distribution-level market").
- The opening drop cap visibly reads **"THe"**. Change the small text after the drop-cap T from "He" to "he."
- "It updates each participant's bid ... and repeatedly solve the market-clearing problem" -> "These algorithms update each participant's bid ... and repeatedly solve the market-clearing problem."
- "the impact of staircase bidding strategy" -> "the impact of the staircase-bidding strategy" or "the impacts of staircase-bidding strategies."
- "In addition to that" -> "Moreover" or "In addition."
- "how the deviation of the bidding strategy of player A affects the revenue of player B" -> "how a deviation by player A affects player B's revenue."
- "a first-of-this-kind study on examining" -> "a first-of-its-kind study examining" or, more cautiously, "a systematic study of."
- "Below summarizes our key contributions" -> "The following summarizes our key contributions."
- "two complementary solution methods, a best-response fixed-point iteration solution and a MARL solution have been developed" -> "we develop two complementary solution methods: a best-response fixed-point iteration method and a MARL method."

### Page 2

- "prone to non-stationary or divergent behavior" -> "prone to nonstationarity or divergence."
- "where expressive bidding structures drastically enlarge the action space" -> "because the more expressive bidding structure drastically enlarges the action space."
- "Whether such structural regularization can potentially improve" -> use either "can improve" or "could potentially improve."
- "moving beyond the conventional price-taker assumption to a price-maker paradigm" -> "moving beyond the conventional price-taker assumption toward a price-maker model."
- "Comprehensive case studies are conducted to systematically evaluate staircase versus single-point bidding strategies" -> "Comprehensive case studies systematically compare staircase and single-point bidding."
- Fig. 1: add spaces in equation references: "Eqs. (1)-(4)," "Eqs. (5)-(10)," and "Eqs. (12)-(29)."

### Page 3

- "cleared quantities at each charging and discharging segments" -> "cleared quantities for the charging and discharging segments."
- "takes positive/negative values when discharging/charging" -> "is positive during discharging and negative during charging."
- Use "SoC" consistently rather than alternating among "SoC," "SOC," and "state of charge."

### Page 4

- After (31), capitalize the new sentence: "At such an equilibrium..."
- "forecasting errors of load and renewables" -> "load and renewable-generation forecast errors."
- "Preliminaries on POMG and MARL are omitted" -> "Preliminaries on POMGs and MARL are omitted."
- Clarify that the active-power component of the DLMP, rather than the full DLMP, is used for compensation.

### Page 5

- "and with \(\alpha^{ch}\ge0\)" -> "and satisfy \(\alpha^{ch}\ge0\)."
- "The staircase bidding enlarges each agent's action space" -> "The staircase-bidding formulation enlarges each agent's action space."
- "Each solution \(a^*\) ... constitutes an expert demonstration \(\omega^{exp}=\{...\}\)" is internally inconsistent: \(a^*\) is an action, whereas the displayed \(\omega^{exp}\) is a full trajectory. Use "Each converged trajectory constitutes one expert demonstration."
- If \(H_{pre}\) counts passes over a static imitation dataset, call them "epochs," not "episodes."

### Page 6

- "becomes searching for the optimal actor parameters" -> "becomes the problem of finding the optimal actor parameters."
- "To illustrate how ... performs against" -> "To compare ... with."
- "robustness of the bidding strategy" -> "robustness of the bidding strategies."
- "test-bed system" -> "testbed."
- "IEEE 33-bus feeder model provided by OpenDSS" may be misleading because OpenDSS is software. Identify the exact feeder data source and cite it.

### Page 7

- "To serve the need of pre-training" -> "To support pre-training."
- "expert demonstrations ... have been obtained" -> "expert demonstrations ... were obtained."
- \(\Omega^{exp}\in\mathbb{R}^{12\times|\omega^{exp}|}\) is not a clear or natural type declaration for a dataset of trajectories. Define it as a set with 12 elements and state each tensor's dimensions separately.
- "The following provides a detailed description of each function block" -> "The following paragraphs describe each functional block."
- "obtained based on the MARL solution" -> "obtained using the MARL solution."
- Use sentence case consistently in subsection headings; "Single-Point Bidding Benchmark" currently differs from nearby headings.

### Page 8

- "As the time evolves" -> "As time progresses."
- "forming short and consecutive 'charge-then-discharge' cycles" -> "forming short, consecutive charge-discharge cycles."
- "the state of charge evolves smoothly" -> "the SoC evolves smoothly."
- "as evidenced by Fig. 5(h)" is repeated in adjacent sentences; condense.
- In Figs. 5 and 6 captions, write "Subplots (a)-(g)" and "Subplot (h)," not "Subplots a)-g)" and "Subplot h)."
- Consider identifying which seven of the eight ESSs are shown and why the eighth is omitted.

### Page 9

- "improve the ESS revenue compared to" -> "increase ESS revenue compared with."
- "its analysis is based on fixed market prices" -> "that analysis assumes fixed market prices."
- "the staircase bidding" -> "staircase bidding."
- "In addition to that" -> "Additionally."
- "helps ESS to be dispatched" -> "helps ESSs be dispatched."
- Write equation ranges consistently as "(26)-(27)" or, preferably in IEEE prose, "(26) and (27)."

### Page 10

- "quantity-price breakpoint" -> "quantity-price breakpoint" is acceptable, but use the same "price-quantity" order used elsewhere.
- "under the single-point bidding" -> "under single-point bidding."
- "As the ESS grows" -> "As the number of ESSs grows."
- "the GS-FPI becomes inherently difficult to converge" -> "GS-FPI becomes increasingly difficult to converge."
- "iteration numbers" -> "numbers of iterations."
- Define why Fig. 10's horizontal axis is "Number of test scenarios," whether the statistics are cumulative, and whether scenario ordering affects the curves.

### Page 11

- "Number of the batteries and size of the system increase scale..." -> "The number of batteries and the system size increase the scale of the proposed EPEC in three ways."
- "To examine scalability ... we add an IEEE 118-bus test case" -> "To evaluate performance on a larger system, we also consider an IEEE 118-bus case."
- Give the Fig. 11 reward unit. If it is daily profit, label the axis accordingly.
- "the staircase bidding curve" -> "the staircase bid curve."
- "More detailed modeling ... will also be considered for future work" -> "Future work will also incorporate more detailed models of distribution-side resources."

### Page 12: references and biographies

- Apply IEEE capitalization consistently to journal titles, for example "Energy Policy" and "IEEE Transactions on Power Systems."
- Reference [17] should identify the document type, such as a Ph.D. dissertation, rather than listing only Stanford University.
- Standardize online-reference formatting and accessed-date style.
- "electricity markets design" in Mengmeng Cai's biography -> "electricity market design."
- "the market integration scheme for distributed energy resources" -> "market-integration schemes for distributed energy resources" or "the market integration of distributed energy resources."
- Consider replacing "in the P.R.C." in Xiangyu Zhang's biography with "in China."

## Consistency edits throughout

- Use one term consistently: "staircase bidding," "staircase-bidding strategy" when adjectival, and "staircase bid curve." Avoid switching among "staircase bid," "step," and "staircase strategy" without definition.
- Use "single-point bidding" without an article in general statements; avoid "the single-point bidding."
- Distinguish revenue \(F^{rev}\), degradation cost, SoC penalty, and profit \(F^{ES}\). Several result sections say "revenue" while the abstract and objective discuss "profitability." State exactly which quantity Tables II and IV report.
- Standardize "GS-FPI" versus "GS–FPI" and hyphen/en-dash use.
- Standardize "nonconverged" versus "non-converged."
- Use "ESSs" as the plural in prose, not "ESS."
- Avoid causal words such as "confirms," "proves," and "as a result" when the evidence is correlational or based on a representative scenario. Prefer "is consistent with," "suggests," or "is associated with."
- Replace novelty claims such as "novel," "first-of-its-kind," and "superior" unless they are carefully scoped and supported by the literature review.

## Recommended revision order

1. Resolve the timing/equivalence issue between the sequential MARL environment and the day-ahead EPEC.
2. Qualify and validate the Nash-equilibrium claim.
3. Redefine or rename the market-efficiency metric.
4. Redesign the paired comparisons and robustness experiment.
5. Fully specify the single-point benchmark and reproducibility details.
6. Apply the page-level language corrections and a final reference-format pass.
