# Synthetic dataset: 9 tables across crm (2 tables), shop (5 tables), inventory (2 tables)

Deterministic generation with seed 20240817: identical output for identical metadata, any partition count.

## Source: crm

### crm_contacts

Root entity, 25,000 rows.

- **contact_id** (int64): primary key
- **state** (string): mostly SP (41.7%), RJ (12.8%), MG (12.0%) (27 categories) and 24 more
- **created_at** (timestamp): between 2015-01-01T00:43:47 and 2018-09-30T22:56:08
- **segment** (string): mostly consumer (78.1%), small_business (14.9%), enterprise (7.0%) (3 categories)
- **marketing_opt_in** (bool): mostly False (64.9%), True (35.1%) (2 categories)

### crm_tickets

Child of `crm_contacts`: on average 0.45 crm_tickets rows per crm_contacts row (up to 8).

- **ticket_id** (int64): primary key
- **contact_id** (int64): reference to the `crm_contacts` row
- **channel** (string): mostly email (36.9%), chat (27.0%), phone (20.5%) (4 categories) and 1 more
- **category** (string): mostly delivery_issue (30.5%), product_question (21.9%), return_request (16.0%) (6 categories) and 3 more
- **priority** (string): mostly low (44.9%), medium (35.6%), high (14.6%) (4 categories) and 1 more
- **opened_at** (timestamp): happens typically 104.6 days (up to ~774.4 days) after `crm_contacts.created_at`
- **resolved_at** (timestamp): happens typically 22.1 h (up to ~10.2 days) after `opened_at` ; 8.0% null
- **csat_score** (int64): mostly 5 (39.6%), 4 (30.1%), 3 (15.5%) (5 categories) and 2 more ; 25.0% null

Event flow: `crm_contacts.created_at` → `opened_at` → `resolved_at`

## Source: shop

### customers

Child of `crm_contacts`: each crm_contacts row has one customers row with probability 60%.

- **customer_id** (int64): primary key
- **contact_id** (int64): reference to the `crm_contacts` row
- **customer_state** (string): inherited from `crm_contacts.state` (master data — always identical to the parent's value)

### orders

Child of `customers`: on average 1 orders rows per customers row (up to 11).

- **order_id** (int64): primary key
- **customer_id** (int64): reference to the `customers` row
- **order_status** (string): mostly delivered (97.0%), shipped (1.2%), canceled (0.7%) (7 categories) and 4 more
- **order_purchase_timestamp** (timestamp): between 2016-09-13T15:24:19 and 2018-10-16T20:16:02
- **order_approved_at** (timestamp): happens typically 21.3 min (up to ~28 h) after `order_purchase_timestamp` ; 0.2% null
- **order_delivered_carrier_date** (timestamp): happens typically 1.8 days (up to ~15.3 days) after `order_approved_at` ; 1.8% null
- **order_delivered_customer_date** (timestamp): happens typically 7.1 days (up to ~41.4 days) after `order_delivered_carrier_date` ; 3.0% null
- **order_estimated_delivery_date** (timestamp): happens typically 23.2 days (up to ~53 days) after `order_purchase_timestamp`

Event flow: `order_purchase_timestamp` → `order_approved_at` → `order_delivered_carrier_date` → `order_delivered_customer_date`
Event flow: `order_purchase_timestamp` → `order_estimated_delivery_date`

### order_items

Child of `orders`: on average 1.1 order_items rows per orders row (up to 23).

- **order_item_id** (int64): primary key
- **order_id** (int64): reference to the `orders` row
- **price** (float64): skewed, median ≈ 76, typical range 30 – 190
- **freight_value** (float64): skewed, median ≈ 17, typical range 8.4 – 33
- **shipping_limit_date** (timestamp): happens typically 6 days (up to ~13.7 days) after `orders.order_purchase_timestamp`
- **product_id** (int64): reference into `inv_products` — popularity-ranked over 9361 items; most popular ≈ 1.1% of picks, top 10 ≈ 4.7%

Correlations (basket): price ↔ freight_value at r = 0.45

Event flow: `orders.order_purchase_timestamp` → `shipping_limit_date`

### order_payments

Child of `orders`: on average 1 order_payments rows per orders row (up to 33).

- **payment_id** (int64): primary key
- **order_id** (int64): reference to the `orders` row
- **payment_type** (string): mostly credit_card (73.4%), boleto (19.5%), voucher (5.8%) (4 categories) and 1 more
- **payment_installments** (int64): mostly 1 (51.0%), 2 (11.9%), 3 (9.9%) (19 categories) and 16 more
- **payment_value** (float64): skewed, median ≈ 99, typical range 38 – 260

Correlations (payment): payment_installments ↔ payment_value at r = 0.42

### order_reviews

Child of `orders`: on average 1 order_reviews rows per orders row (up to 5).

- **review_id** (int64): primary key
- **order_id** (int64): reference to the `orders` row
- **review_score** (int64): mostly 5 (57.6%), 4 (19.1%), 1 (11.9%) (5 categories) and 2 more
- **review_creation_date** (timestamp): happens typically 10.9 days (up to ~39.2 days) after `orders.order_purchase_timestamp`

Event flow: `orders.order_purchase_timestamp` → `review_creation_date`

## Source: inventory

### inv_products

Root entity, 9,361 rows.

- **product_id** (int64): primary key
- **category** (string): mostly bed_bath_table (9.2%), sports_leisure (8.9%), health_beauty (7.6%) (71 categories) and 68 more
- **weight_g** (float64): skewed, median ≈ 820, typical range 210 – 3200
- **length_cm** (float64): skewed, median ≈ 27, typical range 17 – 43
- **height_cm** (float64): skewed, median ≈ 13, typical range 5.7 – 28
- **width_cm** (float64): skewed, median ≈ 21, typical range 13 – 33
- **photos_qty** (int64): mostly 1 (48.5%), 2 (19.5%), 3 (11.6%) (19 categories) and 16 more

### inv_shipments

Child of `orders`: each orders row has one inv_shipments row with probability 98%.

- **shipment_id** (int64): primary key
- **order_id** (int64): reference to the `orders` row
- **warehouse** (string): mostly SP-01 (55.2%), SP-02 (20.0%), RJ-01 (14.7%) (4 categories) and 1 more
- **carrier** (string): mostly correios (62.3%), jadlog (17.8%), azul_cargo (12.1%) (4 categories) and 1 more
- **created_at** (timestamp): happens typically 7.6 h (up to ~1.6 days) after `orders.order_purchase_timestamp`
- **picked_at** (timestamp): happens typically 13.7 h (up to ~2.3 days) after `created_at`
- **handed_over_at** (timestamp): happens typically 5.6 h (up to ~33.7 h) after `picked_at`

Event flow: `orders.order_purchase_timestamp` → `created_at` → `picked_at` → `handed_over_at`

## Privacy

This document contains only fitted statistical parameters (distribution shapes, correlations, cardinalities, temporal delays) — no source records, row-level values, or identifiers are stored in the metadata itself.
This document could be regenerated with differential privacy by re-running `verisynth fit --epsilon <budget>` against the source data, which perturbs each released statistic with Laplace noise before it is written here.
