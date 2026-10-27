# Team Project Charter — Group ___

**Members:**  
**Date:**  
**Repository:**  

Keep the completed charter to approximately one page. Replace every prompt; do not merely delete it.

## Problem formulation

In one paragraph, define:

- the unit of input (page, line, detected box, or character) at each relevant stage;
- the supervision NomNaOCR supplies and the character-level supervision that is missing;
- the output the team must construct; and
- why direct closed-set character classification is risky for historical Nôm.

**Our formulation:** …

## Goals, non-goals, and evidence of success

| ID | Goal (specific and testable) | Measure / acceptance criterion | Owner |
| --- | --- | --- | --- |
| G1 | | | |
| G2 | | | |
| G3 | | | |

**Non-goals for this project:** …

## Roles and working agreement

Roles create accountability, not silos: every member should understand the end-to-end pipeline and every critical task needs a reviewer or backup.

| Member | Primary role and owned artifacts | Reviewer / backup duty |
| --- | --- | --- |
| | Data and evaluation | |
| | Detection and page geometry | |
| | Radical-based recognition / IDS | |
| | Alignment, integration, and reproducibility | |

For a three-person team, combine two adjacent roles. Record how work is reviewed, how decisions are documented, and when blockers are raised: …

## Risk register

Use `L/M/H` for likelihood and impact. A mitigation is a preventive action; a trigger is observable evidence that the contingency should begin.

| Risk | Likelihood | Impact | Mitigation | Trigger / contingency | Owner |
| --- | --- | --- | --- | --- | --- |
| Dataset/label convention is misunderstood | | | | | |
| Page degradation or layout breaks detection/reading order | | | | | |
| Rare/non-standard characters or IDS coverage limits recognition | | | | | |
| Pseudo-label errors reinforce themselves across iterations | | | | | |
| Compute, storage, or schedule is insufficient | | | | | |
| Team dependency or integration failure | | | | | |

## Immediate actions

| Action due before Week 3 | Owner | Evidence when complete |
| --- | --- | --- |
| | | |
| | | |
| | | |
