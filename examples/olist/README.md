# Olist example

## What this is

An end-to-end verisynth example built from the **Olist Brazilian E-Commerce**
public dataset. It shows the full fit -> generate -> validate pipeline on a
real, eleven-table relational dataset spanning **five source systems**
(shop, CRM, inventory, web, EDI) with categorical, numeric, and temporal
(delay-chain) columns, Gaussian-copula correlations, a fitted zipf popularity
dimension reference, and JSON/XML **document synthesis** (schema-shaped and
schema-less), rather than the synthetic data in `examples/retail.yaml`.

Files:

- `prepare_data.py` -- downsamples/cleans the raw Kaggle CSVs into the
  Parquet inputs `verisynth fit` expects (the "shop" source).
- `prepare_crm_data.py` -- synthesizes a second source, a CRM system, from
  the shop sample (see "Two sources" below).
- `prepare_inventory_data.py` -- builds a third source, a product inventory
  system, from the raw product catalog plus the shop sample (see "Third
  source: product inventory" below).
- `prepare_docs_data.py` -- synthesizes two document-oriented sources, a
  web storefront export and an EDI shipment-message feed, as row-for-row
  mirrors of `orders` / `inv_shipments` (see "Document sources" below).
- `schemas/` -- the JSON Schemas (`web_order.schema.json` +
  `web_common.schema.json`, linked by `$ref`) and XSDs
  (`edi_shipment.xsd` includes `edi_common.xsd`) that shape the generated
  JSON/XML documents (docs/ARCHITECTURE.md §11).
- `skeleton.yaml` -- structural metadata (tables, keys, cardinalities,
  copula groups, temporal anchors) with placeholder distribution
  parameters, to be filled in by `verisynth fit`.
- `metadata.olist.yaml` -- the **fitted** metadata, produced by running
  `verisynth fit` against the committed sample. Committed as the expected
  output artifact; `tests/test_olist_integration.py` asserts that refitting
  the committed sample reproduces it.
- `EXPLAIN.md` -- plain-language Markdown rendering of `metadata.olist.yaml`,
  produced by `verisynth explain` (see "Explaining the metadata in plain
  language" below).
- `data/*.parquet` -- a committed, deterministic sample: 15,000 shop
  customers (`customers.parquet`, `orders.parquet`, `order_items.parquet`
  -- now including a real `product_id` column --, `order_payments.parquet`,
  `order_reviews.parquet`) plus 25,000 synthesized CRM contacts
  (`crm_contacts.parquet`, `crm_tickets.parquet`) plus a real
  ~9,361-product inventory catalog and its synthesized shipment orders
  (`inv_products.parquet`, `inv_shipments.parquet`) plus the two synthesized
  document-source mirrors (`web_orders.parquet`, `edi_shipments.parquet`) --
  see "Attribution & license" below -- small enough (~4.9 MB total) to keep
  in the repo so the integration tests always run.

## Attribution & license

Source: [Olist Brazilian E-Commerce Public Dataset](
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle,
uploaded by Olist), licensed **CC BY-NC-SA 4.0**. Mirror used to fetch the
raw CSVs for this example: https://github.com/spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist

The files under `data/` are **not** the original dataset: they are a
15,000-customer deterministic subsample (see "How to regenerate" below),
committed here purely for this repository's own testing/demo purposes, and
carry the same CC BY-NC-SA 4.0 license and attribution requirement as the
source. Do not redistribute this sample (or metadata fitted from it) for
commercial purposes without complying with the license's NonCommercial and
ShareAlike terms; consult the license before any other reuse.

## Schema mapping

| Olist entity | verisynth table | Notes |
|---|---|---|
| `olist_customers_dataset.customer_unique_id` | `customers.customer_id` | The entity key. Olist's `customer_id` in this file is 1:1 with a single order, so it is *not* a stable per-customer id; `customer_unique_id` is. |
| `olist_customers_dataset.customer_state` | `customers.customer_state` | Categorical. |
| `olist_orders_dataset` (joined to customers via `customer_id`) | `orders` | child of `customers`; FK is `customer_unique_id`, renamed `customer_id`. |
| `olist_order_items_dataset` | `order_items` | child of `orders`; one row per line item. |
| `olist_order_payments_dataset` | `order_payments` | child of `orders`; a single order can have multiple payment rows (installments/split payments). |
| `olist_order_reviews_dataset` | `order_reviews` | child of `orders`. |
| `olist_order_items_dataset.product_id` | `order_items.product_id` | the raw string hash, kept as real data in `data/order_items.parquet`; in `skeleton.yaml` it becomes a **dimension reference** (`reference: inv_products`, fitted `zipf{a, n}`) rather than a plain column -- see "Third source: product inventory" below. |
| `olist_products_dataset` (joined to `product_category_name_translation.csv`) | `inv_products` | the real master product catalog, keyed off the distinct `product_id`s in the shop sample's `order_items`; root of the **inventory source** -- see "Third source: product inventory" below. |
| `olist_sellers_dataset`, `olist_geolocation_dataset` | *(not modeled)* | sellers are shared across many orders (many-to-many via order_items, unlike the single-owner product catalog) and geolocation keys off zip-code prefix, neither of which fits the tree-shaped parent/child (or single dimension-reference) model. Modeling them would require multi-parent / many-to-many support that is out of scope for this example. |
| *(synthesized, `prepare_crm_data.py`)* | `crm_contacts`, `crm_tickets` | a second **CRM source**, root of the whole tree; see "Two sources" below. |
| *(synthesized, `prepare_inventory_data.py`)* | `inv_shipments` | shipment orders created by the inventory system from shop orders; child of `orders`; see "Third source: product inventory" below. |
| *(synthesized, `prepare_docs_data.py`)* | `web_orders`, `edi_shipments` | document-oriented mirrors of `orders` / `inv_shipments`, rendered as schema-shaped JSON and XML; see "Document sources" below. |

## Data cleaning applied

- `order_items.freight_value` is floored at `0.01`: a small number of
  free-shipping line items have `freight_value == 0`, which breaks the
  fitter's lognormal-selection rule (`min > 0` required) and would produce a
  degenerate log(0).
- `order_payments` rows with `payment_value <= 0` are dropped: a handful of
  `not_defined` payment-type rows and zero-value voucher rows in the raw
  data are not meaningful payment amounts.
- `order_reviews.review_creation_date` is anchored on `orders.order_purchase_timestamp`
  (rather than, say, `order_delivered_customer_date`) since that is the only
  anchor guaranteed to be non-null for every order in the skeleton's
  temporal DAG.
- `inv_products.category` is `"unknown"` where `product_category_name` is
  null or has no row in `product_category_name_translation.csv`. The raw
  catalog also has a single fully-null row (no category, dimensions, or
  photo count) among the sampled products; its `weight_g`/`length_cm`/
  `height_cm`/`width_cm` are filled with the sample median and
  `photos_qty` with `0` (no `null_rate` is declared for these columns in
  `skeleton.yaml`, so a real ~0.01% missingness rate isn't worth modeling).
- Negative observed delays (an event timestamp earlier than its anchor, a
  handful of data-entry inconsistencies in the raw timestamps) are clamped
  to zero by `verisynth.fit._fit_temporal` before fitting the delay
  distribution (existing, unmodified fitter behavior -- see
  `docs/ARCHITECTURE.md` §7).

## Two sources: CRM as the customer master

This example models **two source systems** sharing one entity tree (see
`docs/ARCHITECTURE.md` §8, the "master-source pattern"):

- **`crm` source** -- `crm_contacts` (root, 25,000 rows) is the customer
  master: every contact's `state`, `segment`, `marketing_opt_in`, and
  `created_at` are authoritative here. `crm_tickets` is its child (support
  tickets, `poisson{lam: 0.45}` per contact), temporally anchored on the
  owning contact's `created_at`.
- **`shop` source** -- `customers` is a **child of `crm_contacts`**, not a
  root: only a fraction of CRM contacts ever became shop customers, modeled
  as `cardinality: bernoulli{p: 0.6}` (fit exactly as `15000 / 25000` from
  the sample -- "a fraction `p` of master records exist downstream", per
  §8). Its `contact_id` column is `generator: parent_key` (which CRM
  contact this shop customer is), and critically its `customer_state`
  column is `generator: "parent:state"` -- **inherited**, not
  independently generated. `orders`, `order_items`, `order_payments`, and
  `order_reviews` hang off `customers` exactly as before.

Because inherited columns are *copied* from the parent's already-generated
value rather than re-sampled, `shop.customers.customer_state` and
`crm.crm_contacts.state` can never disagree for a linked pair -- not
approximately, but by construction, row for row, across any number of
partitions or workers. `tests/test_olist_integration.py::test_master_data_consistency`
checks this with a DuckDB join over the generated output and asserts zero
mismatches (not a statistical tolerance).

The CRM source itself doesn't exist in the raw Olist data -- there is no
real CRM to model. `prepare_crm_data.py` synthesizes it deterministically
from the shop sample: it takes the 15,000 real customers as contacts who
are also shop customers (`state` copied from their real `customer_state`),
adds `--leads` (default 10,000) purely-CRM leads with states drawn from
the real population's empirical state distribution, and generates
`created_at`/`segment`/`marketing_opt_in`/tickets from authored
ground-truth constants (`SEGMENTS`, `CHANNELS`, `CATEGORIES`,
`PRIORITIES`, `CSAT_PROBS`, etc., at the top of the module) via
`numpy.random.default_rng(seed)`. This is a round-trip fidelity demo in
itself: `verisynth fit` refits `crm_contacts`/`crm_tickets`'s categorical
distributions, cardinality, and temporal delays from this synthesized
sample, and the fitted parameters land within ~0.01-0.02 of the authored
constants (see `tests/test_olist_integration.py::test_fitted_crm_categorical_fidelity`
and neighboring tests).

Output layout after `verisynth generate` reflects the source split
(`source:` routes output, per §6/§8 -- it doesn't change generation
semantics); see "Third source: product inventory" below for the third
(`inventory`) directory.

## Third source: product inventory

A third source system, **inventory**, is layered on via
`prepare_inventory_data.py` (TASK CARD 14, docs/ARCHITECTURE.md §8, the
second cross-source mechanism -- dimension references -- alongside the
master-source/child pattern used for CRM):

- **`inv_products`** (root, `source: inventory`) is the **master product
  catalog** -- and unlike `crm_contacts`, this one is **real data**, not
  synthesized: the distinct `product_id`s (9,361 of them) that appear in
  the shop sample's `order_items.parquet`, joined against the raw
  `olist_products_dataset.csv` and `product_category_name_translation.csv`
  to attach `category` (English name; `"unknown"` when untranslated or
  missing), `weight_g`, `length_cm`, `height_cm`, `width_cm`, and
  `photos_qty`.
- `shop.order_items` gains a real `product_id` column (the raw string hash)
  in `data/order_items.parquet`. In `skeleton.yaml` it is declared with
  `reference: inv_products` and an integer `zipf{a, n}` distribution: the
  fitter (`_fit_reference_zipf`, docs/ARCHITECTURE.md §7) discards the
  *identity* of which product each line item points to and instead fits
  only the shape of the popularity profile (`a`, MLE over a deterministic
  grid covering `a >= 0` -- the finite zipfian is well-defined down to
  `a = 0`, uniform over ranks, docs/ARCHITECTURE.md §2) against the
  referenced table's row count (`n = 9361`). In this sample the real
  per-product popularity is close to flat -- 9,361 distinct products across
  17,547 line items, top product ~0.47% of items -- so the fitted `a`
  lands well below 1 (~0.6), not at some `a > 1` "long-tail" floor. This is
  a **star-schema dimension reference**, not a parent/child relationship --
  `order_items` doesn't become a child of `inv_products`, and every value
  the engine samples is clipped to `[0, n-1]` by construction, so
  referential integrity holds without a join at generation time (checked
  anyway by `validate_dataset`'s reference-integrity SQL).
- **`inv_shipments`** (child of `orders`, `source: inventory`) models
  shipment orders the inventory system creates *from* shop orders -- the
  "child-of-another-source's-facts" pattern (§8): the shop `orders` row is
  leading, and `cardinality: bernoulli{p}` fits the fact that a shipment
  exists only for orders whose `order_status` is `delivered` or `shipped`
  (`prepare_inventory_data.py` emits exactly one shipment row per such
  order, none otherwise -- a 0-or-1-per-order structure that
  `verisynth fit` recovers as `bernoulli`, per §7). Its `order_id` column
  is `generator: parent_key` (which shop order this shipment fulfills), and
  `created_at` is **temporally anchored on the parent order's
  `order_purchase_timestamp`** with a strictly-positive lognormal delay --
  so a shipment can never be created before (or at the same instant as) its
  order was placed. `picked_at` and `handed_over_at` chain further delays
  off `created_at`/`picked_at` within the same table.

`tests/test_olist_integration.py::test_shipments_order_leading_and_strictly_later`
checks this end to end: every generated `inv_shipments` row joins to a real
`orders` row (zero orphans) and `created_at > order_purchase_timestamp`
strictly, for every row (not a statistical tolerance -- a hard invariant of
the temporal-delay engine, docs/ARCHITECTURE.md §5, since delays are
clamped at `>= 0` and the fitted lognormal has no zero-mass point).
`test_master_feeds_shop_product_reference` checks the dimension-reference
side: every synthetic `order_items.product_id` resolves to a generated
`inv_products` row, and the synthetic top-ranked product's popularity share
tracks the *real* committed sample's top-product share (a genuine
real-data-fidelity check, not just self-consistency with the fitted pmf it
was derived from) within 0.02 absolute -- both are well under 2%.

Output layout now has five per-source directories (document files sit next
to the Parquet partitions of their source):

```
{out}/crm/crm_contacts/part-00000.parquet, ...
{out}/crm/crm_tickets/part-00000.parquet, ...
{out}/crm/crm_tickets.jsonl              <- schema-less JSON Lines feed
{out}/shop/customers/part-00000.parquet, ...
{out}/shop/orders/part-00000.parquet, ...
{out}/shop/order_items/part-00000.parquet, ...
{out}/shop/order_payments/part-00000.parquet, ...
{out}/shop/order_reviews/part-00000.parquet, ...
{out}/inventory/inv_products/part-00000.parquet, ...
{out}/inventory/inv_shipments/part-00000.parquet, ...
{out}/web/web_orders/part-00000.parquet, ...
{out}/web/web_orders.json                <- shaped by schemas/web_order.schema.json
{out}/edi/edi_shipments/part-00000.parquet, ...
{out}/edi/edi_shipments.xml              <- shaped by schemas/edi_shipment.xsd
```

## Document sources: web (JSON) and EDI (XML)

Two further sources exist only as **document exports** (docs/ARCHITECTURE.md
§11) layered on the relational tree via `prepare_docs_data.py`:

- **`web_orders`** (`source: web`, child of `orders`) is the storefront's
  own order-event export: exactly one event per shop order
  (`cardinality: bernoulli{p: 1.0}`, fit from the mirror sample). `status`
  and `placed_at` are **inherited** (`generator: parent:order_status` /
  `parent:order_purchase_timestamp`), so the JSON export can never disagree
  with the relational shop source; `device` and `utm_source` are authored
  web-only attributes (constants at the top of `prepare_docs_data.py`, fit
  back within ~0.01 like the CRM ones). Its `format:` block renders the
  table as `web/web_orders.json`: a JSON **array** shaped by
  `schemas/web_order.schema.json` -- leaf property names match columns, and
  the nested `attribution` object (a `$ref` into
  `schemas/web_common.schema.json`, exercising multi-schema resolution)
  groups the flat `device`/`utm_source` columns.
- **`edi_shipments`** (`source: edi`, child of `inv_shipments`) is the EDI
  shipment-notification feed: one message per inventory shipment order,
  with `warehouse`/`carrier`/`dispatched_at` inherited from the
  `inv_shipments` row and an authored `service_level`. Its `format:` block
  renders `edi/edi_shipments.xml`: `<shipments>` wrapping one `<shipment>`
  per row, shaped by `schemas/edi_shipment.xsd` (which `xs:include`s
  `schemas/edi_common.xsd` for the named `RoutingType` -- the nested
  `<routing>` element groups `warehouse` + `carrier`).
- **`crm_tickets`** additionally demonstrates the **schema-less** path: a
  bare `format: {kind: jsonl}` renders the flat JSON Lines feed
  `crm/crm_tickets.jsonl` with one object per row, no schema involved.

Documents are rendered from the canonical Parquet partitions with DuckDB,
ordered by primary key, so each file is byte-identical for any
`--partitions` count. `verisynth validate` checks that every declared
document exists and its record count matches the Parquet data, and since
the scanner also *reads* `.json`/`.jsonl`/`.xml`, a directory of generated
documents can itself be scanned (`verisynth scan --input ...`) -- nested
objects/elements flatten back to the leaf-named flat columns they were
shaped from.

## How to regenerate from full data

```bash
# 1. Download the raw CSVs (Kaggle original or the mirror above) into a
#    local directory, e.g. ./olist-raw/ -- must contain at least:
#    olist_customers_dataset.csv, olist_orders_dataset.csv,
#    olist_order_items_dataset.csv, olist_order_payments_dataset.csv,
#    olist_order_reviews_dataset.csv, olist_products_dataset.csv,
#    product_category_name_translation.csv

# 2. Prepare the deterministic shop sample (or omit --sample-customers, or
#    pass 0, to keep every customer):
python examples/olist/prepare_data.py \
    --csv-dir ./olist-raw \
    --out examples/olist/data \
    --sample-customers 15000

# 3. Synthesize the CRM source from the shop sample just written:
python examples/olist/prepare_crm_data.py \
    --data-dir examples/olist/data \
    --out examples/olist/data \
    --leads 10000

# 4. Build the inventory source (real product catalog + synthesized
#    shipment orders) from the raw CSVs and the shop sample:
python examples/olist/prepare_inventory_data.py \
    --csv-dir ./olist-raw \
    --data-dir examples/olist/data \
    --out examples/olist/data

# 5. Build the document-source mirrors (web + EDI) from the shop and
#    inventory samples just written:
python examples/olist/prepare_docs_data.py \
    --data-dir examples/olist/data \
    --out examples/olist/data

# 6. Fit metadata parameters from all five sources (skeleton.table_order()
#    now covers all 11 tables -- crm_*.parquet, inv_*.parquet,
#    web_orders.parquet and edi_shipments.parquet must exist alongside the
#    shop *.parquet files under --input):
verisynth fit --input examples/olist/data \
    -m examples/olist/skeleton.yaml \
    -o examples/olist/metadata.olist.yaml

# 7. Generate a synthetic dataset (writes the Parquet partitions AND the
#    crm_tickets.jsonl / web_orders.json / edi_shipments.xml documents):
verisynth generate -m examples/olist/metadata.olist.yaml -o /tmp/olist-synth --partitions 2

# 8. Validate it (includes the document files):
verisynth validate -m examples/olist/metadata.olist.yaml -o /tmp/olist-synth
```

## Explaining the metadata in plain language

`verisynth explain` renders any metadata document as human-readable Markdown
(structure, distributions, correlations, temporal chains, cross-source
relationships, privacy posture) -- useful for reviewers who don't read the
DSL:

```bash
verisynth explain -m examples/olist/metadata.olist.yaml -o examples/olist/EXPLAIN.md
```

The committed `examples/olist/EXPLAIN.md` is that command's output for this
example's fitted metadata.

## What is preserved

- Marginal distributions: `customer_state`, `order_status`, `payment_type`,
  `payment_installments`, and `review_score` categorical frequencies (e.g.
  `SP` ~42% of customers, `delivered` ~97% of orders, `review_score=5`
  ~57-58% of reviews); `price` and `freight_value` lognormals; `payment_value`
  lognormal.
- Cross-column correlation: the `basket` Gaussian copula between `price` and
  `freight_value` (fitted Spearman-derived correlation ~0.43-0.45) and the
  `payment` copula between `payment_installments` and `payment_value`
  (~0.40-0.45).
- Structural cardinalities: orders-per-customer (~1.03), items-per-order
  (~1.13), payments-per-order (~1.05), reviews-per-order (~1.00), all fit as
  Poisson.
- The temporal delay chain `order_purchase_timestamp -> order_approved_at ->
  order_delivered_carrier_date -> order_delivered_customer_date`, plus
  `order_estimated_delivery_date` and `review_creation_date` anchored on
  purchase time, and `shipping_limit_date` anchored on the parent order's
  purchase time -- including each event's observed null rate.
- The real product catalog's `category` frequencies and `weight_g`/
  `length_cm`/`height_cm`/`width_cm`/`photos_qty` marginals
  (`inv_products`), and the popularity *shape* with which shop
  `order_items` reference the catalog (fitted zipf `a`, docs/ARCHITECTURE.md
  §2/§7) -- though not which specific product is popular (identity is
  discarded by design). The inventory system's shipment-order structure:
  which orders get a shipment at all (fitted `bernoulli{p}`, matching the
  `delivered`/`shipped` `order_status` share), `warehouse`/`carrier`
  frequencies, and the `created_at -> picked_at -> handed_over_at` delay
  chain anchored on the order's purchase time.
- The document sources' authored attributes (`device`/`utm_source` on the
  web export, `service_level` on the EDI feed) and -- by construction, not
  statistically -- the row-for-row agreement of every inherited field
  (`status`, `placed_at`, `warehouse`, `carrier`, `dispatched_at`) between
  the JSON/XML documents and the relational tables they mirror.

## What is not preserved

- Seller identities and geolocation below the state level (see "Schema
  mapping" above for why).
- The dependency between `order_status` and which delivery-chain timestamps
  are null (e.g. a `canceled` order is far more likely to be missing
  `order_delivered_customer_date` than a `delivered` one). The skeleton
  models each timestamp's null rate independently (`null_rate` per column),
  so this conditional structure is only approximated by the columns'
  marginal null rates.
- Seasonality / trend in order volume and timestamps: `order_purchase_timestamp`
  is fit as `datetime_uniform` over the observed min/max range, which
  flattens Olist's real growth curve and calendar seasonality (e.g. the
  Black Friday spike) into a uniform distribution.
- The true shape of the `order_approved_at` delay: this column is
  *bimodal in log-space* -- roughly the fastest 60-70% of orders are
  auto-approved within minutes, while the rest take hours to days (manual
  review / boleto payment confirmation). The fitter (per
  `docs/ARCHITECTURE.md` §7) handles the ~1.4% exact-zero delays by fitting
  a robust lognormal (`mu = median(log d)`, `sigma = std(log d)`) on the
  strictly-positive subset; the median-based `mu` anchors the *typical*
  delay exactly (synthetic/sample median delay ratio ~1.0), so most
  approval delays are faithful. Zero-delay mass itself is not reproduced
  (the fitted distribution is continuous, so no synthetic delay is exactly
  0), and because a single lognormal cannot represent two modes, the upper
  quartile/tail of this column is compressed relative to the real data
  (sample Q75 ~15h vs. synthetic Q75 ~1.5h, i.e. `exp(mu + 0.674*sigma)`).
  A mixture-of-lognormals (or similar multi-component) delay kind that
  could capture both the fast and slow approval modes is future work.

## How to add differential privacy

Pass `--epsilon` to `verisynth fit` to release DP-perturbed parameters
instead (Laplace mechanism, total budget split evenly across released
statistics -- see `docs/ARCHITECTURE.md` §7). Numeric marginal columns
require a `clamp: [lo, hi]` bound declared in the skeleton (DP sensitivity
is computed from the clamp range), so add e.g.:

```yaml
price:
  type: float64
  distribution: {kind: lognormal, mu: 4.3, sigma: 1.0}
  clamp: [0.01, 7000.0]
freight_value:
  type: float64
  distribution: {kind: lognormal, mu: 2.8, sigma: 0.6}
  clamp: [0.01, 500.0]
payment_value:
  type: float64
  distribution: {kind: lognormal, mu: 4.6, sigma: 0.9}
  clamp: [0.01, 15000.0]
```

to `skeleton.yaml` for each numeric distribution column (`price`,
`freight_value`, `payment_value`; categorical and `datetime_uniform`
columns don't need `clamp`), then run:

```bash
verisynth fit --input examples/olist/data \
    -m examples/olist/skeleton.yaml \
    -o examples/olist/metadata.olist.dp.yaml \
    --epsilon 1.0 --dp-seed 0
```
