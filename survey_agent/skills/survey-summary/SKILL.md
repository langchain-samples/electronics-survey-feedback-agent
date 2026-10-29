---
name: survey-summary
description: Format a customer survey theme summary for survey owners or a Slack/Teams channel. Use this skill whenever you draft or post a survey-feedback summary.
---

# Survey Summary Skill

## Format

```
**Survey pulse: <date range>** (<N> responses, NPS <score>)

**Top themes**
1. **<Theme name>** (<count> responses, avg score <x>): <one-line description>
   > "<verbatim quote>" (R-xxxx)
   > "<verbatim quote>" (R-xxxx)
2. ...

**What's working**
- <positive theme with one quote>

**Suggested next steps**
- <one action per top theme, owner if known>
```

## Rules

- 3-5 themes, ordered by impact on NPS (most negative first).
- Quotes are copied exactly from the survey `comment` field and cited by `response_id`.
- Numbers come from `get_survey_stats` or the theme counts, never from estimates.
- Redact emails and phone numbers as `[redacted]`.
- Under 300 words. No emojis in channel posts.
