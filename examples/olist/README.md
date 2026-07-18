# Olist example

## What this is

An end-to-end verisynth example built from the **Olist Brazilian E-Commerce**
public dataset. It shows the full fit -> generate -> validate pipeline on a
real, five-table relational dataset with categorical, numeric, and temporal
(delay-chain) columns and Gaussian-copula correlations, rather than the
synthetic data in `examples/retail.yaml`.

Files:

- `prepare_data.py` -- downsamples/cleans the raw Kaggle CSVs into the
  Parquet inputs `verisynth fit` expects (the "shop" source).
- `prepare_crm_data.py` -- synthesizes a second source, a CRM system, from
  the shop sample (see "Two sources" below).
- `skeleton.yaml` -- structural metadata (tables, keys, cardinalities,
  copula groups, temporal anchors) with placeholder distribution
  parameters, to be filled in by `verisynth fit`.
- `metadata.olist.yaml` -- the **fitted** metadata, produced by running
  `verisynth fit` against the committed sample. Committed as the expected
  output artifact; `tests/test_olist_integration.py` asserts that refitting
  the committed sample reproduces it.
- `data/*.parquet` -- a committed, deterministic sample: 15,000 shop
  customers (`customers.parquet`, `orders.parquet`, `order_items.parquet`,
  `order_payments.parquet`, `order_reviews.parquet`) plus 25,000 synthesized
  CRM contacts (`crm_contacts.parquet`, `crm_tickets.parquet`) -- see
  "Attribution & license" below -- small enough (~3.3 MB total) to keep in
  the repo so the integration tests always run.

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
| `olist_products_dataset`, `olist_sellers_dataset`, `olist_geolocation_dataset` | *(not modeled)* | verisynth's current metadata DSL supports a single-parent tree per table (`role: child` + one `parent`); products/sellers are shared across many orders (many-to-many via order_items) and geolocation keys off zip-code prefix, neither of which fits the tree-shaped parent/child model. Modeling them would require multi-parent / reference-table support that is out of scope for this example. |
| *(synthesized, `prepare_crm_data.py`)* | `crm_contacts`, `crm_tickets` | a second **CRM source**, root of the whole tree; see "Two sources" below. |

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
semantics):

```
{out}/crm/crm_contacts/part-00000.parquet, ...
{out}/crm/crm_tickets/part-00000.parquet, ...
{out}/shop/customers/part-00000.parquet, ...
{out}/shop/orders/part-00000.parquet, ...
{out}/shop/order_items/part-00000.parquet, ...
{out}/shop/order_payments/part-00000.parquet, ...
{out}/shop/order_reviews/part-00000.parquet, ...
```

## How to regenerate from full data

```bash
# 1. Download the raw CSVs (Kaggle original or the mirror above) into a
#    local directory, e.g. ./olist-raw/ -- must contain at least:
#    olist_customers_dataset.csv, olist_orders_dataset.csv,
#    olist_order_items_dataset.csv, olist_order_payments_dataset.csv,
#    olist_order_reviews_dataset.csv

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

# 4. Fit metadata parameters from both sources (skeleton.table_order() now
#    covers all 7 tables -- crm_contacts.parquet/crm_tickets.parquet must
#    exist alongside the shop *.parquet files under --input):
verisynth fit --input examples/olist/data \
    -m examples/olist/skeleton.yaml \
    -o examples/olist/metadata.olist.yaml

# 5. Generate a synthetic dataset:
verisynth generate -m examples/olist/metadata.olist.yaml -o /tmp/olist-synth --partitions 2

# 6. Validate it:
verisynth validate -m examples/olist/metadata.olist.yaml -o /tmp/olist-synth
```

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

## What is not preserved

- The product catalog, seller identities, and geolocation below the
  state level (see "Schema mapping" above for why).
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
