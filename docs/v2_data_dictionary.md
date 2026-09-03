# V2 数据字典（自动生成）

由 `scripts/generate_data_dictionary.py` 从 `research_v2/migrations/*.dos`
冻结合约渲染。迁移变更后必须重新生成本文件。

迁移版本：001–014；表数量：47。

## active_model_pointer

- 来源：迁移 004；生产者：registry.active_pointer
- 说明：consumer active model pointer + switch audit
- 业务键：`-`
- 分区列：`effective_at`；排序列：`consumer_id, environment, pointer_version, effective_at`

| 列名 | 类型 |
|---|---|
| consumer_id | SYMBOL |
| environment | SYMBOL |
| model_id | STRING |
| model_version | INT |
| pointer_version | LONG |
| effective_at | TIMESTAMP |
| previous_model_id | STRING |
| previous_model_version | INT |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## data_quality_result

- 来源：迁移 001；生产者：quality.engine DQ run
- 说明：blocking DQ check outcomes
- 业务键：`-`
- 分区列：`created_at`；排序列：`object_name, check_name, created_at`

| 列名 | 类型 |
|---|---|
| run_id | UUID |
| object_name | SYMBOL |
| check_name | SYMBOL |
| scope_name | STRING |
| severity | SYMBOL |
| status | SYMBOL |
| observed_value | STRING |
| expected_rule | STRING |
| sample_json | STRING |
| created_at | TIMESTAMP |

## dataset_manifest

- 来源：迁移 004；生产者：datasets.manifest
- 说明：content-addressed dataset identity
- 业务键：`-`
- 分区列：`created_at`；排序列：`dataset_id, created_at`

| 列名 | 类型 |
|---|---|
| dataset_id | STRING |
| fold_id | SYMBOL |
| manifest_json | STRING |
| artifact_uri | STRING |
| row_count | LONG |
| manifest_hash | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## experiment_run

- 来源：迁移 004；生产者：training.search experiment audit
- 说明：trial lease/audit records
- 业务键：`-`
- 分区列：`started_at`；排序列：`experiment_id, started_at`

| 列名 | 类型 |
|---|---|
| experiment_id | STRING |
| model_family | SYMBOL |
| model_config_hash | STRING |
| dataset_id | STRING |
| seed | LONG |
| status | SYMBOL |
| artifact_uri | STRING |
| started_at | TIMESTAMP |
| finished_at | TIMESTAMP |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_observation_clean

- 来源：迁移 003；生产者：factor materialization
- 说明：cleaned admitted factor observations
- 业务键：`-`
- 分区列：`signal_date`；排序列：`factor_id, ts_code, signal_date`

| 列名 | 类型 |
|---|---|
| signal_date | DATE |
| ts_code | SYMBOL |
| factor_id | SYMBOL |
| factor_version | INT |
| value | DOUBLE |
| observation_date | DATE |
| available_date | DATE |
| report_period | DATE |
| stale_days | INT |
| dedup_resolution | SYMBOL |
| source_revision_id | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_observation_clean_latest

- 来源：迁移 008；生产者：compact pipeline
- 说明：latest compact factor observations
- 业务键：`-`
- 分区列：`signal_date`；排序列：`factor_id, ts_code, signal_date`

| 列名 | 类型 |
|---|---|
| batch_id | LONG |
| signal_date | DATE |
| ts_code | SYMBOL |
| factor_id | SYMBOL |
| factor_version | INT |
| value | DOUBLE |
| observation_date | DATE |
| available_date | DATE |
| report_period | DATE |
| stale_days | INT |
| dedup_resolution | SYMBOL |
| source_revision_id | STRING |

## factor_pair_redundancy

- 来源：迁移 014；生产者：scripts.record_regime_window_quality
- 说明：pairwise median abs Spearman redundancy per window
- 业务键：`-`
- 分区列：`as_of_date`；排序列：`factor_id_left, factor_id_right, window_id, as_of_date`

| 列名 | 类型 |
|---|---|
| as_of_date | DATE |
| factor_id_left | SYMBOL |
| factor_id_right | SYMBOL |
| window_id | SYMBOL |
| median_abs_spearman | DOUBLE |
| valid_day_count | INT |
| pair_class | SYMBOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_quality_grade

- 来源：迁移 014；生产者：scripts.record_regime_window_quality
- 说明：four-dimension quality grade per factor
- 业务键：`-`
- 分区列：`as_of_date`；排序列：`factor_id, as_of_date`

| 列名 | 类型 |
|---|---|
| as_of_date | DATE |
| factor_id | SYMBOL |
| factor_group | SYMBOL |
| grade | SYMBOL |
| total_score | INT |
| score_train_predictive | INT |
| score_oos_validation | INT |
| score_regime_robustness | INT |
| score_uniqueness | INT |
| train_icir | DOUBLE |
| validation_icir | DOUBLE |
| regime_min_official_icir | DOUBLE |
| max_similarity_edge | DOUBLE |
| grade_rules_version | INT |
| evidence_path | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_quality_latest

- 来源：迁移 012；生产者：factor quality latest pipeline
- 说明：latest quality snapshot
- 业务键：`-`
- 分区列：`effective_date`；排序列：`factor_id, effective_date`

| 列名 | 类型 |
|---|---|
| factor_id | SYMBOL |
| source_factor_id | SYMBOL |
| effective_date | DATE |
| decision | SYMBOL |
| model_eligible | BOOL |
| monitoring_required | BOOL |
| reason_code | STRING |
| market_scope | SYMBOL |
| label_id | SYMBOL |
| train_start | DATE |
| train_end | DATE |
| validation_start | DATE |
| validation_end | DATE |
| frozen_direction | INT |
| train_valid_days | INT |
| train_signed_rank_ic | DOUBLE |
| train_icir | DOUBLE |
| train_directional_share | DOUBLE |
| annual_gate_years | INT |
| train_max_similarity | DOUBLE |
| validation_valid_days | INT |
| validation_signed_rank_ic | DOUBLE |
| validation_icir | DOUBLE |
| validation_directional_share | DOUBLE |
| validation_max_similarity | DOUBLE |
| evidence_path | STRING |
| evidence_commit | STRING |

## factor_quality_snapshot

- 来源：迁移 003；生产者：factor quality pipeline
- 说明：factor quality evidence snapshot
- 业务键：`-`
- 分区列：`as_of_date`；排序列：`factor_id, factor_version, as_of_date`

| 列名 | 类型 |
|---|---|
| as_of_date | DATE |
| factor_id | SYMBOL |
| factor_version | INT |
| scope_id | STRING |
| eligibility_status | SYMBOL |
| metrics_json | STRING |
| rules_version | INT |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_quarantine

- 来源：迁移 003；生产者：factor admission quarantine
- 说明：rejected factor rows audit
- 业务键：`-`
- 分区列：`observation_date`；排序列：`factor_id, ts_code, observation_date`

| 列名 | 类型 |
|---|---|
| observation_date | DATE |
| ts_code | SYMBOL |
| factor_id | SYMBOL |
| factor_version | INT |
| reason | SYMBOL |
| source_row_ids_json | STRING |
| distinct_values_json | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_rank_ic_daily

- 来源：迁移 006；生产者：materializers.factor_evaluation
- 说明：factor RankIC daily
- 业务键：`-`
- 分区列：`signal_date`；排序列：`factor_id, factor_version, signal_date`

| 列名 | 类型 |
|---|---|
| signal_date | DATE |
| factor_id | SYMBOL |
| factor_version | INT |
| factor_name | STRING |
| rank_ic | DOUBLE |
| sample_count | LONG |
| evaluation_scope | STRING |
| label_id | SYMBOL |
| label_version | INT |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_rank_ic_summary

- 来源：迁移 006；生产者：materializers.factor_evaluation
- 说明：factor RankIC summary
- 业务键：`-`
- 分区列：`as_of_date`；排序列：`factor_id, factor_version, as_of_date`

| 列名 | 类型 |
|---|---|
| as_of_date | DATE |
| factor_id | SYMBOL |
| factor_version | INT |
| factor_name | STRING |
| evaluation_scope | STRING |
| valid_day_count | INT |
| mean_rank_ic | DOUBLE |
| rank_ic_std | DOUBLE |
| icir | DOUBLE |
| positive_day_share | DOUBLE |
| mean_sample_count | DOUBLE |
| label_id | SYMBOL |
| label_version | INT |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_registry

- 来源：迁移 003；生产者：factor admission registry
- 说明：admitted factor identities/versions
- 业务键：`-`
- 分区列：`created_at`；排序列：`factor_id, factor_version, created_at`

| 列名 | 类型 |
|---|---|
| factor_id | SYMBOL |
| factor_version | INT |
| display_name | STRING |
| factor_type | SYMBOL |
| frequency | SYMBOL |
| expected_direction | SYMBOL |
| source_database | STRING |
| source_table | SYMBOL |
| source_value_column | SYMBOL |
| availability_policy | SYMBOL |
| max_staleness_calendar_days | INT |
| unit | SYMBOL |
| definition_hash | STRING |
| owner | SYMBOL |
| status | SYMBOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_repair_audit

- 来源：迁移 009；生产者：factor repair pipeline
- 说明：repair action audit
- 业务键：`-`
- 分区列：`as_of_date`；排序列：`factor_id, logical_partition, as_of_date`

| 列名 | 类型 |
|---|---|
| as_of_date | DATE |
| logical_partition | STRING |
| factor_id | SYMBOL |
| factor_version | INT |
| source_table | SYMBOL |
| source_factor_name | STRING |
| repair_method | SYMBOL |
| source_rows | LONG |
| candidate_rows | LONG |
| exact_duplicate_groups | LONG |
| exact_duplicate_rows | LONG |
| conflict_groups | LONG |
| nonfinite_rows | LONG |
| overlap_finite_rows | LONG |
| fillable_rows | LONG |
| filled_rows | LONG |
| added_rows | LONG |
| remaining_nonfinite_rows | LONG |
| mismatch_rows | LONG |
| max_abs_diff | DOUBLE |
| max_rel_diff | DOUBLE |
| pit_violation_rows | LONG |
| unresolved_rows | LONG |
| decision_status | SYMBOL |
| rules_version | INT |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_repair_candidate

- 来源：迁移 010；生产者：factor repair pipeline
- 说明：repair candidates
- 业务键：`-`
- 分区列：`observation_date`；排序列：`factor_id, ts_code, observation_date`

| 列名 | 类型 |
|---|---|
| observation_date | DATE |
| ts_code | SYMBOL |
| factor_id | SYMBOL |
| factor_version | INT |
| factor_name | STRING |
| candidate_value | DOUBLE |
| source_value | DOUBLE |
| repair_method | SYMBOL |
| reason | SYMBOL |
| candidate_hash | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_repair_quarantine

- 来源：迁移 009；生产者：factor repair pipeline
- 说明：repair quarantine
- 业务键：`-`
- 分区列：`observation_date`；排序列：`factor_id, ts_code, reason, observation_date`

| 列名 | 类型 |
|---|---|
| observation_date | DATE |
| ts_code | SYMBOL |
| factor_id | SYMBOL |
| factor_version | INT |
| factor_name | STRING |
| reason | SYMBOL |
| source_value | DOUBLE |
| candidate_value | DOUBLE |
| repair_method | SYMBOL |
| candidate_hash | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_set

- 来源：迁移 003；生产者：factor set freezing
- 说明：frozen factor sets
- 业务键：`-`
- 分区列：`created_at`；排序列：`factor_set_id, rules_version, created_at`

| 列名 | 类型 |
|---|---|
| factor_set_id | STRING |
| selection_scope_id | STRING |
| rules_version | INT |
| member_count | INT |
| status | SYMBOL |
| content_hash | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_set_member

- 来源：迁移 003；生产者：factor set freezing
- 说明：factor set membership
- 业务键：`-`
- 分区列：`created_at`；排序列：`factor_set_id, factor_id, factor_version, created_at`

| 列名 | 类型 |
|---|---|
| factor_set_id | STRING |
| factor_id | SYMBOL |
| factor_version | INT |
| feature_order | INT |
| frozen_direction | INT |
| selection_reason | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_similarity

- 来源：迁移 003；生产者：candidate pair similarity
- 说明：factor pair similarity evidence
- 业务键：`-`
- 分区列：`as_of_date`；排序列：`factor_id_left, factor_id_right, as_of_date`

| 列名 | 类型 |
|---|---|
| as_of_date | DATE |
| scope_id | STRING |
| factor_id_left | SYMBOL |
| factor_id_right | SYMBOL |
| median_daily_abs_spearman | DOUBLE |
| valid_days | INT |
| status | SYMBOL |
| cluster_id | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## factor_window_quality

- 来源：迁移 014；生产者：scripts.record_regime_window_quality
- 说明：signed IC metrics per factor/window/universe (pre-training + train + validation)
- 业务键：`-`
- 分区列：`as_of_date`；排序列：`factor_id, window_id, universe_id, as_of_date`

| 列名 | 类型 |
|---|---|
| as_of_date | DATE |
| factor_id | SYMBOL |
| factor_version | INT |
| factor_group | SYMBOL |
| window_id | SYMBOL |
| universe_id | SYMBOL |
| valid_day_count | INT |
| signed_mean_rank_ic | DOUBLE |
| signed_icir | DOUBLE |
| directional_day_share | DOUBLE |
| evidence_path | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## fundamental_pit_snapshot

- 来源：迁移 011；生产者：fundamental snapshot pipeline
- 说明：fundamental PIT data
- 业务键：`-`
- 分区列：`signal_date`；排序列：`metric_id, ts_code, signal_date`

| 列名 | 类型 |
|---|---|
| signal_date | DATE |
| ts_code | SYMBOL |
| metric_id | SYMBOL |
| value | DOUBLE |
| report_period | DATE |
| ann_date | DATE |
| available_date | DATE |
| source_revision_id | STRING |
| stale_days | INT |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## label_daily

- 来源：迁移 003；生产者：materializers.labels
- 说明：primary forward-open 5d labels
- 业务键：`signal_date, ts_code, label_id, label_version`
- 分区列：`signal_date`；排序列：`ts_code, label_id, label_version, signal_date`

| 列名 | 类型 |
|---|---|
| signal_date | DATE |
| ts_code | SYMBOL |
| label_id | SYMBOL |
| label_version | INT |
| entry_date | DATE |
| exit_date | DATE |
| label_available_date | DATE |
| raw_return | DOUBLE |
| rank_pct | DOUBLE |
| winsorized_return | DOUBLE |
| zscore_return | DOUBLE |
| neutral_residual | DOUBLE |
| label_valid | BOOL |
| invalid_reason | SYMBOL |
| entry_realized_tradable | BOOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## label_daily_latest

- 来源：迁移 008；生产者：compact pipeline
- 说明：latest compact labels
- 业务键：`-`
- 分区列：`signal_date`；排序列：`ts_code, label_id, label_version, signal_date`

| 列名 | 类型 |
|---|---|
| batch_id | LONG |
| signal_date | DATE |
| ts_code | SYMBOL |
| label_id | SYMBOL |
| label_version | INT |
| entry_date | DATE |
| exit_date | DATE |
| label_available_date | DATE |
| raw_return | DOUBLE |
| rank_pct | DOUBLE |
| winsorized_return | DOUBLE |
| zscore_return | DOUBLE |
| neutral_residual | DOUBLE |
| label_valid | BOOL |
| invalid_reason | SYMBOL |
| entry_realized_tradable | BOOL |

## latest_manifest

- 来源：迁移 008；生产者：compact pipeline
- 说明：latest compact manifests
- 业务键：`-`
- 分区列：`published_at`；排序列：`object_name, logical_partition, published_at`

| 列名 | 类型 |
|---|---|
| object_name | SYMBOL |
| logical_partition | STRING |
| batch_id | LONG |
| source_object | SYMBOL |
| source_data_version | STRING |
| published_at | TIMESTAMP |
| row_count | LONG |
| business_hash | STRING |
| status | SYMBOL |

## locked_test_audit

- 来源：迁移 004；生产者：registry.locked_test
- 说明：2026 locked test audit trail
- 业务键：`-`
- 分区列：`event_at`；排序列：`model_id, model_version, event_at`

| 列名 | 类型 |
|---|---|
| audit_id | UUID |
| model_id | STRING |
| model_version | INT |
| action | SYMBOL |
| preregistration_hash | STRING |
| capability_hash | STRING |
| actor | SYMBOL |
| approver | SYMBOL |
| consumed | BOOL |
| event_at | TIMESTAMP |
| details_json | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## market_daily_clean

- 来源：迁移 002；生产者：materializers.market
- 说明：cleaned daily market + history features
- 业务键：`trade_date, ts_code`
- 分区列：`trade_date`；排序列：`ts_code, trade_date`

| 列名 | 类型 |
|---|---|
| trade_date | DATE |
| ts_code | SYMBOL |
| raw_open | DOUBLE |
| raw_high | DOUBLE |
| raw_low | DOUBLE |
| raw_close | DOUBLE |
| volume | LONG |
| amount | DOUBLE |
| adj_factor | DOUBLE |
| adjusted_open | DOUBLE |
| valid_trade_day | BOOL |
| valid_days_20 | INT |
| adv20 | DOUBLE |
| listing_age_trading_days | INT |
| source_flags | STRING |
| dedup_resolution | SYMBOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## market_quarantine

- 来源：迁移 005；生产者：materializers.market (quarantine)
- 说明：invalid adjustment-factor audit
- 业务键：`-`
- 分区列：`trade_date`；排序列：`ts_code, field_name, trade_date`

| 列名 | 类型 |
|---|---|
| trade_date | DATE |
| ts_code | SYMBOL |
| field_name | SYMBOL |
| reason | SYMBOL |
| raw_value | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## metric_daily

- 来源：迁移 004；生产者：evaluation metric materializer
- 说明：daily evaluation metrics
- 业务键：`-`
- 分区列：`signal_date`；排序列：`experiment_id, fold_id, metric_name, signal_date`

| 列名 | 类型 |
|---|---|
| experiment_id | STRING |
| fold_id | SYMBOL |
| signal_date | DATE |
| metric_name | SYMBOL |
| metric_value | DOUBLE |
| sample_count | INT |
| valid | BOOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## metric_summary

- 来源：迁移 004；生产者：evaluation metric materializer
- 说明：summary evaluation metrics
- 业务键：`-`
- 分区列：`created_at`；排序列：`experiment_id, fold_id, metric_scope, metric_name, created_at`

| 列名 | 类型 |
|---|---|
| experiment_id | STRING |
| fold_id | SYMBOL |
| metric_scope | SYMBOL |
| metric_name | SYMBOL |
| metric_value | DOUBLE |
| status | SYMBOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## migration_lock

- 来源：迁移 001；生产者：db.migrations runner
- 说明：advisory lock for migration runs
- 业务键：`-`
- 分区列：`acquired_at`；排序列：`lock_name, acquired_at`

| 列名 | 类型 |
|---|---|
| lock_name | SYMBOL |
| owner_run_id | UUID |
| acquired_at | TIMESTAMP |
| expires_at | TIMESTAMP |

## model_oos_rank_ic_daily

- 来源：迁移 007；生产者：materializers.oos
- 说明：model OOS RankIC daily evidence
- 业务键：`signal_date, evaluation_id`
- 分区列：`signal_date`；排序列：`evaluation_id, model_id, signal_date`

| 列名 | 类型 |
|---|---|
| signal_date | DATE |
| evaluation_id | STRING |
| model_id | SYMBOL |
| fold_id | SYMBOL |
| role | SYMBOL |
| rank_ic | DOUBLE |
| sample_count | LONG |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |
| eligible_count | LONG |
| valid_label_count | LONG |
| coverage | DOUBLE |
| prediction_unique_count | LONG |
| prediction_std | DOUBLE |
| rank_ic_valid | BOOL |

## model_oos_rank_ic_summary

- 来源：迁移 007；生产者：materializers.oos
- 说明：model OOS RankIC summary evidence
- 业务键：`-`
- 分区列：`as_of_date`；排序列：`evaluation_id, model_id, as_of_date`

| 列名 | 类型 |
|---|---|
| as_of_date | DATE |
| evaluation_id | STRING |
| model_id | SYMBOL |
| fold_id | SYMBOL |
| role | SYMBOL |
| valid_day_count | INT |
| mean_rank_ic | DOUBLE |
| rank_ic_std | DOUBLE |
| icir | DOUBLE |
| positive_day_share | DOUBLE |
| mean_sample_count | DOUBLE |
| metrics_json | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## model_registry

- 来源：迁移 004；生产者：registry.model_registry
- 说明：model lifecycle status machine
- 业务键：`-`
- 分区列：`transitioned_at`；排序列：`model_id, model_version, transitioned_at`

| 列名 | 类型 |
|---|---|
| model_id | STRING |
| model_version | INT |
| model_family | SYMBOL |
| status | SYMBOL |
| artifact_uri | STRING |
| artifact_hash | STRING |
| evidence_json | STRING |
| previous_status | SYMBOL |
| transitioned_at | TIMESTAMP |
| transitioned_by | SYMBOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## pipeline_run

- 来源：迁移 001；生产者：runs.audit pipeline_run context manager
- 说明：run lifecycle audit
- 业务键：`-`
- 分区列：`started_at`；排序列：`pipeline_name, started_at`

| 列名 | 类型 |
|---|---|
| run_id | UUID |
| pipeline_name | SYMBOL |
| status | SYMBOL |
| data_version | STRING |
| config_hash | STRING |
| code_commit | STRING |
| started_at | TIMESTAMP |
| finished_at | TIMESTAMP |
| error_class | SYMBOL |
| error_message | STRING |
| environment_json | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## prediction_daily

- 来源：迁移 004；生产者：inference.publisher
- 说明：published daily predictions
- 业务键：`-`
- 分区列：`signal_date`；排序列：`model_id, model_version, ts_code, signal_date`

| 列名 | 类型 |
|---|---|
| model_id | STRING |
| model_version | INT |
| signal_date | DATE |
| ts_code | SYMBOL |
| raw_score | DOUBLE |
| score | DOUBLE |
| rank | INT |
| percentile_rank | DOUBLE |
| universe_id | SYMBOL |
| universe_version | INT |
| inference_manifest_id | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## published_version

- 来源：迁移 001；生产者：db.publish publisher
- 说明：visibility pointer per object/partition
- 业务键：`object_name, logical_partition, published_at`
- 分区列：`published_at`；排序列：`object_name, published_at`

| 列名 | 类型 |
|---|---|
| object_name | SYMBOL |
| logical_partition | STRING |
| data_version | STRING |
| previous_data_version | STRING |
| published_at | TIMESTAMP |
| run_id | UUID |

## schema_migration

- 来源：迁移 001；生产者：db.migrations runner
- 说明：audit of applied migrations
- 业务键：`schema_version`
- 分区列：`applied_at`；排序列：`applied_at, schema_version`

| 列名 | 类型 |
|---|---|
| schema_version | INT |
| name | STRING |
| checksum | STRING |
| applied_at | TIMESTAMP |
| applied_by | SYMBOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## security_master_scd

- 来源：迁移 002；生产者：materializers.security_master
- 说明：point-in-time security SCD
- 业务键：`ts_code, valid_from`
- 分区列：`valid_from`；排序列：`ts_code, valid_from`

| 列名 | 类型 |
|---|---|
| ts_code | SYMBOL |
| exchange | SYMBOL |
| board | SYMBOL |
| list_date | DATE |
| delist_date | DATE |
| security_name | STRING |
| is_st | BOOL |
| listing_status | SYMBOL |
| valid_from | DATE |
| valid_to | DATE |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## source_reconciliation

- 来源：迁移 001；生产者：sources reconciliation (Tushare)
- 说明：cross-source reconciliation records
- 业务键：`-`
- 分区列：`trade_date`；排序列：`ts_code, field_name, trade_date`

| 列名 | 类型 |
|---|---|
| trade_date | DATE |
| ts_code | SYMBOL |
| field_name | SYMBOL |
| check_version | INT |
| primary_value | STRING |
| secondary_value | STRING |
| abs_diff | DOUBLE |
| rel_diff | DOUBLE |
| status | SYMBOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## source_watermark

- 来源：迁移 001；生产者：backfill/source adapters
- 说明：source observation watermarks
- 业务键：`-`
- 分区列：`captured_at`；排序列：`source_name, table_name, captured_at`

| 列名 | 类型 |
|---|---|
| source_watermark_id | STRING |
| source_name | SYMBOL |
| table_name | SYMBOL |
| max_observation_date | DATE |
| source_row_count | LONG |
| schema_fingerprint | STRING |
| captured_at | TIMESTAMP |
| run_id | UUID |

## split_definition

- 来源：迁移 004；生产者：datasets.splits
- 说明：frozen split definitions
- 业务键：`-`
- 分区列：`created_at`；排序列：`split_id, fold_id, split_role, created_at`

| 列名 | 类型 |
|---|---|
| split_id | STRING |
| fold_id | SYMBOL |
| split_role | SYMBOL |
| start_date | DATE |
| end_date | DATE |
| purge_trading_days | INT |
| locked | BOOL |
| rules_hash | STRING |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## tradability_daily

- 来源：迁移 002；生产者：materializers.tradability
- 说明：signal-date tradability rules
- 业务键：`trade_date, ts_code`
- 分区列：`trade_date`；排序列：`ts_code, trade_date`

| 列名 | 类型 |
|---|---|
| trade_date | DATE |
| ts_code | SYMBOL |
| is_st | BOOL |
| is_suspended | BOOL |
| is_listed | BOOL |
| listing_age_trading_days | INT |
| valid_days_20 | INT |
| adv20 | DOUBLE |
| is_eligible_pre_liquidity | BOOL |
| reason_bitmap | LONG |
| rule_version | INT |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## tradability_daily_latest

- 来源：迁移 008；生产者：compact pipeline
- 说明：latest compact tradability
- 业务键：`-`
- 分区列：`trade_date`；排序列：`ts_code, trade_date`

| 列名 | 类型 |
|---|---|
| batch_id | LONG |
| trade_date | DATE |
| ts_code | SYMBOL |
| is_st | BOOL |
| is_suspended | BOOL |
| is_listed | BOOL |
| listing_age_trading_days | INT |
| valid_days_20 | INT |
| adv20 | DOUBLE |
| is_eligible_pre_liquidity | BOOL |
| reason_bitmap | LONG |
| rule_version | INT |

## universe_daily

- 来源：迁移 002；生产者：universe materialization
- 说明：daily universe membership
- 业务键：`-`
- 分区列：`signal_date`；排序列：`universe_id, universe_version, ts_code, signal_date`

| 列名 | 类型 |
|---|---|
| signal_date | DATE |
| universe_id | SYMBOL |
| universe_version | INT |
| ts_code | SYMBOL |
| is_member | BOOL |
| exclusion_reason | SYMBOL |
| reason_bitmap | LONG |
| adv20_rank | INT |
| membership_rank | INT |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |

## universe_daily_latest

- 来源：迁移 008；生产者：compact pipeline
- 说明：latest compact universe
- 业务键：`-`
- 分区列：`signal_date`；排序列：`universe_id, universe_version, ts_code, signal_date`

| 列名 | 类型 |
|---|---|
| batch_id | LONG |
| signal_date | DATE |
| universe_id | SYMBOL |
| universe_version | INT |
| ts_code | SYMBOL |
| is_member | BOOL |
| exclusion_reason | SYMBOL |
| reason_bitmap | LONG |
| adv20_rank | INT |
| membership_rank | INT |

## universe_definition

- 来源：迁移 002；生产者：universe definitions
- 说明：immutable universe rule versions
- 业务键：`-`
- 分区列：`created_at`；排序列：`universe_id, universe_version, created_at`

| 列名 | 类型 |
|---|---|
| universe_id | SYMBOL |
| universe_version | INT |
| rules_json | STRING |
| rules_hash | STRING |
| status | SYMBOL |
| run_id | UUID |
| data_version | STRING |
| created_at | TIMESTAMP |
| source_watermark_id | STRING |
| record_hash | STRING |
