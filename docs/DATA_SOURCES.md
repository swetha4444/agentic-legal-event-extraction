# Data sources: Free Law Project & CourtListener

This project uses U.S. court data for **role-aware legal event extraction**. We focus on **District court** cases with a **nature of suit** indicator, preferring **Employment** (NOS 442) or **Criminal**, and **Complaint** documents when available; opinions supplement.

## Free Law Project

- **Home:** [Free Law Project](https://www.freelawproject.org/) — *Making the legal ecosystem more equitable and competitive.*
- **CourtListener:** [courtlistener.com](https://www.courtlistener.com/) — search and browse case law, RECAP (PACER) dockets, judges, oral arguments.
- **APIs:** [CourtListener API help](https://www.courtlistener.com/help/api/) — REST APIs, bulk data, replication.

## This pipeline: cluster-first (only cases with court opinion)

We only use cases that have a court opinion. We fetch from the **Clusters API** (case law); each record has a court opinion as **document_text**. Then for each cluster we use its **docket_id** to fetch docket metadata (docket_number, court_id, date_terminated, assigned_to, etc.) and **parties** from the Docket and Parties APIs. If the Parties API is unavailable we use parties from the case caption. Auth: standard CourtListener token.

- **Clusters API** (`GET /api/rest/v4/clusters/`) — we paginate clusters (optionally filter by `nature_of_suit`). For each cluster we fetch the first `sub_opinion` for `plain_text` / `html_with_citations`; that becomes **document_text**. Each cluster has a **docket_id** (one docket per cluster in case law). We get **parties** from the case caption by default.
- **Docket API** (`GET /api/rest/v4/dockets/{id}/`) — for each cluster we call this with that cluster's **docket_id** to get docket metadata: docket_number, court_id, date_terminated, assigned_to, referred_to, cause, etc.
- **Parties API** (`GET /api/rest/v4/parties/?docket={docket_id}`) — we fetch parties for the same docket_id. If the API returns parties we use them; otherwise we keep parties from the case caption. Auth: same CourtListener token (`Authorization: Token <token>`).

## How the data is fetched (cluster-first)

### What is a docket? What is docket_id?

A **docket** is the court’s file for one case: it has a docket number (e.g. `5:98-cv-01772`), parties, filings, and metadata. CourtListener stores every docket with a numeric **docket_id** (e.g. `5354075`). That id is the same everywhere in CourtListener: RECAP (PACER) data and case-law (opinion) data both attach to the same docket table. So one **docket_id** = one case in CourtListener.

### Cluster-first flow

1. We call the **Clusters API** with pagination (and optional `nature_of_suit` filter). Every result is a **cluster**: one opinion (or set of opinions) tied to **one docket**, with a **cluster_id** and **docket_id**.
2. For each cluster we fetch the first **sub_opinion** text → that is **document_text** (and **opinion_text**) for the record.
3. We take the cluster's **docket_id** and call the **Docket API** for that id to get docket_number, court_id, date_terminated, assigned_to, referred_to, cause, etc.).
4. We call the **Parties API** with the same docket_id. If we get a list of parties we use it; otherwise we keep parties parsed from the case caption (e.g. "Plaintiff v. Defendant").
5. We write one JSONL line per cluster with **cluster_id**, **docket_id**, court opinion as **document_text**, and all docket/parties metadata.

So: every row has a **cluster_id** and **docket_id**; **document_text** is always the court opinion; **parties** come from the Parties API or caption.

## API URLs used by this pipeline

All URLs are under the CourtListener REST API base: **`https://www.courtlistener.com/api/rest/v4/`**. Authentication: `Authorization: Token <your_token>`.

| API | URL | What it does | Main params / usage |
|-----|-----|---------------|----------------------|
| **Clusters** | `GET .../clusters/` | Lists **opinion clusters** (case-law): each cluster = one docket + one or more opinions (e.g. main opinion, concurrence). Returns `cluster_id`, `docket_id`, case name, date_filed, nature_of_suit, `sub_opinions` (list of opinion URLs), etc. | Pagination via `next` (cursor in response). Optional: `nature_of_suit` to filter by NOS (e.g. 442). We paginate and optionally filter by NOS; for each cluster we then fetch the first opinion’s text from the **Opinion** URL. |
| **Opinion** | `GET .../opinions/{id}/` | Returns a **single opinion**’s metadata and text. We request `plain_text` and `html_with_citations` fields; we use `plain_text` (or strip HTML) as the court opinion text. | The `id` comes from each cluster’s `sub_opinions` (e.g. `.../opinions/12345/`). Called once per cluster (first sub_opinion) when we want opinion text. |
| **Docket** | `GET .../dockets/{id}/` | Returns **one docket** by numeric id: docket number, court, date_filed, date_terminated, assigned_to, referred_to, cause, and other docket metadata. Does **not** include the list of parties. | `{id}` = the cluster’s `docket_id`. We call this for each cluster to enrich the record with docket metadata. |
| **Parties** | `GET .../parties/?docket={docket_id}` | Returns **parties** for a given docket (e.g. plaintiff, defendant names). May be empty or restricted for some dockets. | `docket={docket_id}` = the cluster’s docket id. We use the returned party list when available; otherwise we keep parties from the case caption. |

**RECAP Search** (`GET .../search/?type=r`) is **not** used by the cluster-first pipeline. It searches RECAP (PACER) dockets by query; the pipeline instead uses only Clusters → Docket → Parties (and Opinion for text).

## RECAP (PACER archive)

- **What it is:** Federal **dockets** and **documents** (complaints, motions, etc.) from PACER. Search results include **parties**, attorneys, firms, assigned judge, referred magistrate, jurisdiction_type, jury_demand, and document snippets.
- **Example search:** [employment, NOS 442, filed before 2022-09-05 — ~45k results](https://www.courtlistener.com/?q=employment&type=r&nature_of_suit=442&filed_before=2022-09-05)
- **APIs:** [Legal Search API](https://www.courtlistener.com/help/api/rest/search/) (type=r); [PACER data APIs](https://www.courtlistener.com/help/api/rest/pacer/) (docket-entries, recap-documents) for full document text — contact CourtListener for access.

## Output (JSONL)

Each line is one case. Common fields: `case_id`, `docket_id`, `cluster_id`, `case_name`, `case_name_full`, `date_filed`, `date_terminated`, `court_id`, `court`, `document_text`, `absolute_url`, `parties`, `attorneys`, `firms`, `assigned_to`, `referred_to`, `jurisdiction_type`, `jury_demand`, `documents` (list with description/snippet per doc). `document_text` is always the court opinion (we only include cases that have an opinion). `parties` come from the Parties API by docket_id or from the case caption.

## Nature of suit (examples)

- **442** — Civil Rights: Jobs (employment)

## Links

- [Free Law Project](https://www.freelawproject.org/)
- [CourtListener](https://www.courtlistener.com/)
- [CourtListener API](https://www.courtlistener.com/help/api/)
- [Case Law API](https://www.courtlistener.com/help/api/rest/case-law/) (clusters, opinions)
- [RECAP Search](https://www.courtlistener.com/?q=employment&type=r&nature_of_suit=442&filed_before=2022-09-05)
