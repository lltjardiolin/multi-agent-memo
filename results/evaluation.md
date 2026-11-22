# Evaluation results (2026-11-22 15:27:11)

Model `gpt-4o-mini` at temperature 0.0 for both systems. Failure = error, non-completed status, or static-rubric fail.

## baseline

| Topic | Status | Body words | Rubric | Turns | Rejections | MCP calls | Tokens | Seconds | Failed | Rubric notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T01 | completed | 284 | FAIL | 1 | 0 | 0 | 1087 | 7.5 | yes | 5 sentence(s) with figures lack a citation, e.g. 'This memo provides an overview of the trends in rooftop solar installations in the Northst' |
| T02 | completed | 270 | FAIL | 1 | 0 | 0 | 1061 | 5.5 | yes | 3 sentence(s) with figures lack a citation, e.g. 'This memo provides an overview of the sales and pricing trends for Solara home batteries i' |
| T03 | completed | 285 | FAIL | 1 | 0 | 0 | 1113 | 5.8 | yes | 2 sentence(s) with figures lack a citation, e.g. "This memo provides an overview of SwiftCart's delivery growth, reliability, and cost struc"; markdown formatting in body |
| T04 | completed | 285 | FAIL | 1 | 0 | 0 | 1100 | 6.7 | yes | 2 sentence(s) with figures lack a citation, e.g. 'In 2025, online grocery shopping in Northstar saw a notable increase in adoption, with 600' |
| T05 | completed | 265 | FAIL | 1 | 0 | 0 | 1046 | 5.4 | yes | 2 sentence(s) with figures lack a citation, e.g. '**To:** [Recipient Name] **From:** [Your Name] **Date:** [Current Date] **Subject:** Asses'; markdown formatting in body |

Failures 5/5 (rate 1.00); total tokens 5407; mean time 6.2 s per topic; mean turns 1.0; rejections 0; est. cost $0

## team

| Topic | Status | Body words | Rubric | Turns | Rejections | MCP calls | Tokens | Seconds | Failed | Rubric notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T01 | escalated_to_human | 264 | pass | 8 | 3 | 3 | 47930 | 46.3 | yes | - |
| T02 | completed | 250 | pass | 8 | 2 | 3 | 49700 | 60.2 | no | - |
| T03 | escalated_to_human | 258 | pass | 8 | 3 | 3 | 64075 | 80.7 | yes | - |
| T04 | escalated_to_human | 255 | pass | 8 | 3 | 3 | 62334 | 76.9 | yes | - |
| T05 | escalated_to_human | 279 | pass | 8 | 3 | 3 | 51117 | 62.5 | yes | - |

Failures 4/5 (rate 0.80); total tokens 275156; mean time 65.3 s per topic; mean turns 8.0; rejections 14; est. cost $0.051544

## team_dual

| Topic | Status | Body words | Rubric | Turns | Rejections | MCP calls | Tokens | Seconds | Failed | Rubric notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T01 | completed | 259 | pass | 6 | 1 | 3 | 39120 | 51.6 | no | - |
| T02 | escalated_to_human | 258 | pass | 8 | 3 | 3 | 53242 | 63.1 | yes | - |
| T03 | escalated_to_human | 266 | pass | 8 | 3 | 3 | 58199 | 67.3 | yes | - |
| T04 | escalated_to_human | 273 | pass | 8 | 3 | 3 | 78668 | 88.5 | yes | - |
| T05 | escalated_to_human | 266 | pass | 8 | 3 | 3 | 58142 | 64.3 | yes | - |

Failures 4/5 (rate 0.80); total tokens 287371; mean time 67.0 s per topic; mean turns 7.6; rejections 13; est. cost $0.053188

**Error amplification (team / baseline):** 0.8 (goal <= 1.0: met)
**Error amplification (team_dual / baseline):** 0.8 (goal <= 1.0: met)

**Stretch cost:** dual reviewer used 287371 tokens vs 275156 single (1.04x), est. $0.053188 vs $0.051544; failures 4 vs 4.
