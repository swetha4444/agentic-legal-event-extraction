# Data sources: Free Law Project & CourtListener

This project uses U.S. court data for **role-aware legal event extraction**. We focus on **District court** cases with a **nature of suit** indicator, preferring **Employment** (NOS 442) or **Criminal**, and **Complaint** documents when available; opinions supplement.

## Free Law Project

- **Home:** [Free Law Project](https://www.freelawproject.org/) — *Making the legal ecosystem more equitable and competitive.*
- **CourtListener:** [courtlistener.com](https://www.courtlistener.com/) — search and browse case law, RECAP (PACER) dockets, judges, oral arguments.
- **APIs:** [CourtListener API help](https://www.courtlistener.com/help/api/) — REST APIs, bulk data, replication.

## This pipeline: CourtListener only

| Source | Content | Auth | Use |
|--------|--------|------|-----|
| **Clusters API** | Case law (opinion clusters) + opinion text | Standard token | Default: clusters + full opinion text per case |
| **RECAP Search** | Federal dockets + doc snippets | May require subscription | `--source recap` for complaints/dockets |

- **Clusters:** `GET /api/rest/v4/clusters/` — then for each cluster fetch first `sub_opinion` for `plain_text` / `html_with_citations`. Same token as curl with `Authorization: Token <token>`.
- **RECAP search:** `GET /api/rest/v4/search/?type=r` — filter by `nature_of_suit=442`, `q=employment`, etc. Returns dockets with up to 3 document snippets.

## RECAP (PACER archive)

- **What it is:** Federal **dockets** and **documents** (complaints, motions, etc.) from PACER.
- **Example search:** [employment, NOS 442, filed before 2022-09-05 — ~45k results](https://www.courtlistener.com/?q=employment&type=r&nature_of_suit=442&filed_before=2022-09-05)
- **APIs:** [Legal Search API](https://www.courtlistener.com/help/api/rest/search/) (type=r); [PACER data APIs](https://www.courtlistener.com/help/api/rest/pacer/) (docket-entries, recap-documents) for full document text — contact CourtListener for access.

## Nature of suit (examples)

- **442** — Civil Rights: Jobs (employment)

## Links

- [Free Law Project](https://www.freelawproject.org/)
- [CourtListener](https://www.courtlistener.com/)
- [CourtListener API](https://www.courtlistener.com/help/api/)
- [Case Law API](https://www.courtlistener.com/help/api/rest/case-law/) (clusters, opinions)
- [RECAP Search](https://www.courtlistener.com/?q=employment&type=r&nature_of_suit=442&filed_before=2022-09-05)
