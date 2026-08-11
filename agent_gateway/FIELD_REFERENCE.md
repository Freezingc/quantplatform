# Database Field Reference

This document is generated from `数据库/daily_task/update_*.py` for agent-side factor generation.
`update_alpha191.py` and `update_factor.py` are skipped because they update factor workflows rather than base database tables.

Use these fields as the source of truth before generating DSL expressions or production `FactorCalculator` drafts.

## Summary

| Table | Update Script | Field Count |
| --- | --- | ---: |
| `balance_sheet` | `update_balance_sheet.py` | 153 |
| `cashflow_table` | `update_cashflow_table.py` | 98 |
| `company_income_table` | `update_company_income.py` | 86 |
| `financial_metrics_table` | `update_financial_metrics_table.py` | 168 |
| `index_component_table` | `update_index_component.py` | 6 |
| `index_daily_data` | `update_index_daily_data.py` | 12 |
| `index_info_table` | `update_index_info_table.py` | 13 |
| `stock_daily_data` | `update_stock_daily_data.py` | 12 |
| `stock_factor_daily` | `update_stock_factor_daily.py` | 262 |
| `stock_info_table` | `update_stock_info_table.py` | 17 |
| `stock_trading_suspension_info` | `update_stock_trading_suspension_info.py` | 5 |
| `stock_used_name` | `update_stock_used_name.py` | 6 |
| `sw_ind_code` | `update_sw_ind_code.py` | 7 |
| `trade_calendar` | `update_trade_calendar.py` | 4 |

## Tables

### `balance_sheet`

- Update script: `数据库/daily_task/update_balance_sheet.py`
- Field count: 153

| Field | Role | Agent Usage |
| --- | --- | --- |
| `acc_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `acc_receivable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `accounts_pay` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `accounts_receiv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `accounts_receiv_bill` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `acct_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `acting_trading_sec` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `acting_uw_sec` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `adv_receipts` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `agency_bus_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `amor_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ann_date` | identifier/date | Join key; do not use as a factor signal. |
| `bond_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `cap_rese` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cash_reser_cb` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cb_borr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cip` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cip_total` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `client_depos` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `client_prov` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `comm_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `comp_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `const_materials` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `contract_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `contract_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `cost_fin_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `debt_invest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `decr_in_disbur` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `defer_inc_non_cur_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `defer_tax_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `defer_tax_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `deferred_inc` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `depos` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `depos_ib_deposits` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `depos_in_oth_bfi` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `depos_oth_bfi` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `depos_received` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `deriv_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `deriv_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `div_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `div_receiv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `end_date` | identifier/date | Join key; do not use as a factor signal. |
| `end_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `estimated_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `f_ann_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fa_avail_for_sale` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fair_value_fin_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `fix_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `fix_assets_total` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `fixed_assets_disp` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `forex_differ` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `goodwill` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `hfs_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `hfs_sales` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `htm_invest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `indem_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `indep_acct_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `indept_acc_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `int_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `int_receiv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `intan_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `inventories` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `invest_as_receiv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `invest_loss_unconf` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `invest_real_estate` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `lending_funds` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `loan_oth_bank` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `loanto_oth_bank_fi` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `long_pay_total` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `lt_amor_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `lt_borr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `lt_eqt_invest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `lt_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `lt_payroll_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `lt_rec` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `minority_int` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `money_cap` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `nca_within_1y` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `non_cur_liab_due_1y` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `notes_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `notes_receiv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oil_and_gas_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `ordin_risk_reser` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `oth_comp_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `oth_cur_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `oth_cur_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `oth_debt_invest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_eqt_tools` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_eqt_tools_p_shr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `oth_nca` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_ncl` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_pay_total` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_rcv_total` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_receiv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `payable_to_reinsurer` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `payables` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `payroll_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ph_invest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ph_pledge_loans` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `pledge_borr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `policy_div_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `prec_metals` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `prem_receiv_adva` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `premium_receiv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `prepayment` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `produc_bio_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `pur_resale_fa` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `r_and_d` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `refund_cap_depos` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `refund_depos` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `reinsur_receiv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `reinsur_res_receiv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `report_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `reser_lins_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `reser_lthins_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `reser_outstd_claims` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `reser_une_prem` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `rr_reins_lins_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `rr_reins_lthins_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `rr_reins_outstd_cla` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `rr_reins_une_prem` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `rsrv_insur_cont` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `sett_rsrv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `sold_for_repur_fa` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `special_rese` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `specific_payables` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `st_bonds_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `st_borr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `st_fin_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `surplus_rese` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `taxes_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `time_deposits` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `total_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `total_cur_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `total_cur_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `total_hldr_eqy_exc_min_int` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `total_hldr_eqy_inc_min_int` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `total_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `total_liab_hldr_eqy` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `total_nca` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `total_ncl` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `total_share` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `trad_asset` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `trading_fl` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `transac_seat_fee` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `treasury_share` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |
| `undistr_porfit` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `update_flag` | raw | Allowed for candidate expressions if coverage is sufficient. |

### `cashflow_table`

- Update script: `数据库/daily_task/update_cashflow_table.py`
- Field count: 98

| Field | Role | Agent Usage |
| --- | --- | --- |
| `amort_intang_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `ann_date` | identifier/date | Join key; do not use as a factor signal. |
| `beg_bal_cash` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `beg_bal_cash_equ` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_cash_equ_beg_period` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_cash_equ_end_period` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_disp_withdrwl_invest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_fr_oth_operate_a` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_fr_sale_sg` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_inf_fr_operate_a` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_paid_for_taxes` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_paid_goods_s` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_paid_invest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_paid_to_for_empl` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_pay_acq_const_fiolta` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_pay_claims_orig_inco` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_pay_dist_dpcp_int_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_prepay_amt_borr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_recp_borrow` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_recp_cap_contrib` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `c_recp_return_invest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `comp_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `conv_copbonds_due_within_1y` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `conv_debt_into_cap` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `credit_impa_loss` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `decr_def_inc_tax_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `decr_deferred_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `decr_inventories` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `decr_oper_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `depr_fa_coga_dpba` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `eff_fx_flu_cash` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `end_bal_cash` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `end_bal_cash_equ` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `end_date` | identifier/date | Join key; do not use as a factor signal. |
| `end_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `f_ann_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fa_fnc_leases` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `finan_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `free_cashflow` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `ifc_cash_incr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `im_n_incr_cash_equ` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `im_net_cashflow_oper_act` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `incl_cash_rec_saims` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `incl_dvd_profit_paid_sc_ms` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `incr_acc_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `incr_def_inc_tax_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `incr_oper_payable` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `invest_loss` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `loss_disp_fiolta` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `loss_fv_chg` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `loss_scr_fa` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `lt_amort_deferred_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_cap_incr_repur` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_cash_flows_fnc_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_cashflow_act` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `n_cashflow_inv_act` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `n_depos_incr_fi` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_disp_subs_oth_biz` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_inc_borr_oth_fi` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_incr_cash_cash_equ` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_incr_clt_loan_adv` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_incr_dep_cbob` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_incr_disp_faas` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_incr_disp_tfa` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_incr_insured_dep` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_incr_loans_cb` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_incr_loans_oth_bank` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_incr_pledge_loan` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_recp_disp_fiolta` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_recp_disp_sobu` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_reinsur_prem` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `net_cash_rece_sec` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `net_dism_capital_add` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `net_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `oth_cash_pay_oper_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_cash_recp_ral_fnc_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_cashpay_ral_fnc_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_loss_asset` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_pay_ral_inv_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oth_recp_ral_inv_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `others` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `pay_comm_insur_plcy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `pay_handling_chrg` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `prem_fr_orig_contr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `proc_issue_bonds` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `prov_depr_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `recp_tax_rends` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `report_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `st_cash_out_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `stot_cash_in_fnc_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `stot_cashout_fnc_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `stot_inflows_inv_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `stot_out_inv_act` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |
| `uncon_invest_loss` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `update_flag` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `use_right_asset_dep` | raw | Allowed for candidate expressions if coverage is sufficient. |

### `company_income_table`

- Update script: `数据库/daily_task/update_company_income.py`
- Field count: 86

| Field | Role | Agent Usage |
| --- | --- | --- |
| `adj_lossgain` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `admin_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ann_date` | identifier/date | Join key; do not use as a factor signal. |
| `ass_invest_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `assets_impair_loss` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `basic_eps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `biz_tax_surchg` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `capit_comstock_div` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `comm_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `comm_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `comp_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `compens_payout` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `compens_payout_refu` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `compr_inc_attr_m_s` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `compr_inc_attr_p` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `comshare_payable_dvd` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `continued_net_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `diluted_eps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `distable_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `distr_profit_shrhder` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `div_payt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ebit` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ebitda` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `end_date` | identifier/date | Join key; do not use as a factor signal. |
| `end_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `f_ann_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fin_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fin_exp_int_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fin_exp_int_inc` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `forex_gain` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fv_value_chg_gain` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `income_tax` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `insur_reser_refu` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `insurance_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `int_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `int_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `invest_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `minority_gain` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_asset_mg_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `n_commis_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `n_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `n_income_attr_p` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `n_oth_b_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `n_oth_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `n_sec_tb_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `n_sec_uw_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `nca_disploss` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `non_oper_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `non_oper_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `oper_cost` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `oper_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `operate_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `oth_b_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `oth_compr_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `other_bus_cost` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `out_prem` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `prem_earned` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `prem_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `prem_refund` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `prfshare_payable_dvd` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `rd_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `reins_cost_refund` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `reins_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `reins_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `report_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `reser_insur_liab` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `revenue` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `sell_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `t_compr_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `total_cogs` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `total_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `total_revenue` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `transfer_housing_imprest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `transfer_oth` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `transfer_surplus_rese` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |
| `undist_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `une_prem_reser` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `update_flag` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `withdra_biz_devfund` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `withdra_legal_pubfund` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `withdra_legal_surplus` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `withdra_oth_ersu` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `withdra_rese_fund` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `workers_welfare` | raw | Allowed for candidate expressions if coverage is sufficient. |

### `financial_metrics_table`

- Update script: `数据库/daily_task/update_financial_metrics_table.py`
- Field count: 168

| Field | Role | Agent Usage |
| --- | --- | --- |
| `adminexp_of_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ann_date` | identifier/date | Join key; do not use as a factor signal. |
| `ar_turn` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `arturn_days` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `assets_to_eqt` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `assets_turn` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `assets_yoy` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `basic_eps_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bps_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ca_to_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `ca_turn` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `capital_rese_ps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `capitalized_to_da` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cash_ratio` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cash_to_liqdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cash_to_liqdebt_withinterest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cfps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cfps_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cogs_of_sales` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `current_exint` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `current_ratio` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `currentdebt_to_debt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `daa` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `debt_to_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `debt_to_eqt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `diluted2_eps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dp_assets_to_eqt` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `dt_eps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dt_eps_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dt_netprofit_yoy` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `dtprofit_to_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `ebit` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ebit_of_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ebit_ps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ebit_to_interest` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ebitda` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ebitda_to_debt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ebt_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `end_date` | identifier/date | Join key; do not use as a factor signal. |
| `eps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `eqt_to_debt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `eqt_to_interestdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `eqt_to_talcapital` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `eqt_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `equity_yoy` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `expense_of_sales` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `extra_item` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fa_turn` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fcfe` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fcfe_ps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fcff` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fcff_ps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `finaexp_of_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fixed_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `gc_of_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `gross_margin` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `grossprofit_margin` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `impai_ttm` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `int_to_talcap` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `interestdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `interst_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `inv_turn` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `invest_capital` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `investincome_of_ebt` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `invturn_days` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `longdeb_to_debt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `longdebt_to_workingcapital` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `n_op_profit_of_ebt` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `nca_to_assets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `netdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `netprofit_margin` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `netprofit_yoy` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `networking_capital` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `non_op_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `noncurrent_exint` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `nop_to_ebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `npta` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ocf_to_debt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ocf_to_interestdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ocf_to_netdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ocf_to_opincome` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `ocf_to_or` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ocf_to_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `ocf_to_shortdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ocf_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ocfps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `op_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `op_of_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `op_to_debt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `op_to_ebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `op_to_liqdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `op_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `opincome_of_ebt` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `or_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `profit_dedt` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `profit_prefin_exp` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `profit_to_gr` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `profit_to_op` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_adminexp_to_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_dt_roe` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_dtprofit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_dtprofit_to_profit` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_eps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_exp_to_sales` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_finaexp_to_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_gc_to_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_gr_qoq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_gr_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_gsprofit_margin` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_impair_to_gr_ttm` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_investincome` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_investincome_to_ebt` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_netprofit_margin` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_netprofit_qoq` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_netprofit_yoy` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_npta` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_ocf_to_or` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_ocf_to_sales` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_op_qoq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_op_to_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_op_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_opincome` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_opincome_to_ebt` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_profit_qoq` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_profit_to_gr` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_profit_yoy` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `q_roe` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_saleexp_to_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_sales_qoq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_sales_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `q_salescash_to_or` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `quick_ratio` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `rd_exp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `retained_earnings` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `retainedps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `revenue_ps` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `roa` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roa2_yearly` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roa_dp` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roa_yearly` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roe` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roe_avg` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roe_dt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roe_waa` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roe_yearly` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roe_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roic` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roic_yearly` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `saleexp_to_gr` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `salescash_to_or` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `surplus_rese_ps` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `tangasset_to_intdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `tangible_asset` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `tangibleasset_to_debt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `tangibleasset_to_netdebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `tax_to_ebt` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `tbassets_to_totalassets` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `total_fa_trun` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `total_revenue_ps` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `tr_yoy` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |
| `turn_days` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `undist_profit_ps` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `update_flag` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `valuechange_income` | financial | Allowed for candidate expressions if coverage is sufficient. |
| `working_capital` | raw | Allowed for candidate expressions if coverage is sufficient. |

### `index_component_table`

- Update script: `数据库/daily_task/update_index_component.py`
- Field count: 6

| Field | Role | Agent Usage |
| --- | --- | --- |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `con_code` | identifier/date | Join key; do not use as a factor signal. |
| `index_code` | identifier/date | Join key; do not use as a factor signal. |
| `trade_date` | identifier/date | Join key; do not use as a factor signal. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |
| `weight` | raw | Allowed for candidate expressions if coverage is sufficient. |

### `index_daily_data`

- Update script: `数据库/daily_task/update_index_daily_data.py`
- Field count: 12

| Field | Role | Agent Usage |
| --- | --- | --- |
| `amount` | volume/liquidity | Allowed for candidate expressions if coverage is sufficient. |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `change` | price | Allowed for candidate expressions if coverage is sufficient. |
| `close` | price | Allowed for candidate expressions if coverage is sufficient. |
| `high` | price | Allowed for candidate expressions if coverage is sufficient. |
| `low` | price | Allowed for candidate expressions if coverage is sufficient. |
| `open` | price | Allowed for candidate expressions if coverage is sufficient. |
| `pct_chg` | price | Allowed for candidate expressions if coverage is sufficient. |
| `pre_close` | price | Allowed for candidate expressions if coverage is sufficient. |
| `trade_date` | identifier/date | Join key; do not use as a factor signal. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |
| `vol` | volume/liquidity | Allowed for candidate expressions if coverage is sufficient. |

### `index_info_table`

- Update script: `数据库/daily_task/update_index_info_table.py`
- Field count: 13

| Field | Role | Agent Usage |
| --- | --- | --- |
| `base_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `base_point` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `category` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `desc` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `exp_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fullname` | metadata | Use for filtering or labeling, not numeric DSL expressions. |
| `index_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `list_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `market` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `name` | metadata | Use for filtering or labeling, not numeric DSL expressions. |
| `publisher` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |
| `weight_rule` | raw | Allowed for candidate expressions if coverage is sufficient. |

### `stock_daily_data`

- Update script: `数据库/daily_task/update_stock_daily_data.py`
- Field count: 12

| Field | Role | Agent Usage |
| --- | --- | --- |
| `amount` | volume/liquidity | Allowed for candidate expressions if coverage is sufficient. |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `change` | price | Allowed for candidate expressions if coverage is sufficient. |
| `close` | price | Allowed for candidate expressions if coverage is sufficient. |
| `high` | price | Allowed for candidate expressions if coverage is sufficient. |
| `low` | price | Allowed for candidate expressions if coverage is sufficient. |
| `open` | price | Allowed for candidate expressions if coverage is sufficient. |
| `pct_chg` | price | Allowed for candidate expressions if coverage is sufficient. |
| `pre_close` | price | Allowed for candidate expressions if coverage is sufficient. |
| `trade_date` | identifier/date | Join key; do not use as a factor signal. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |
| `vol` | volume/liquidity | Allowed for candidate expressions if coverage is sufficient. |

### `stock_factor_daily`

- Update script: `数据库/daily_task/update_stock_factor_daily.py`
- Field count: 262

| Field | Role | Agent Usage |
| --- | --- | --- |
| `adj_factor` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `amount` | volume/liquidity | Allowed for candidate expressions if coverage is sufficient. |
| `asi_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `asi_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `asi_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `asit_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `asit_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `asit_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `atr_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `atr_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `atr_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `bbi_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bbi_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bbi_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bias1_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bias1_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bias1_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bias2_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bias2_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bias2_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bias3_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bias3_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `bias3_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `boll_lower_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `boll_lower_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `boll_lower_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `boll_mid_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `boll_mid_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `boll_mid_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `boll_upper_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `boll_upper_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `boll_upper_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `brar_ar_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `brar_ar_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `brar_ar_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `brar_br_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `brar_br_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `brar_br_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `cci_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `cci_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `cci_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `change` | price | Allowed for candidate expressions if coverage is sufficient. |
| `circ_mv` | valuation/market-cap | Allowed for candidate expressions if coverage is sufficient. |
| `close` | price | Allowed for candidate expressions if coverage is sufficient. |
| `close_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `close_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cr_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cr_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cr_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dfma_dif_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dfma_dif_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dfma_dif_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dfma_difma_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dfma_difma_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dfma_difma_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_adx_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_adx_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_adx_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_adxr_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_adxr_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_adxr_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_mdi_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_mdi_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_mdi_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_pdi_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_pdi_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `dmi_pdi_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `downdays` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dpo_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dpo_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dpo_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `dv_ratio` | valuation/market-cap | Allowed for candidate expressions if coverage is sufficient. |
| `dv_ttm` | valuation/market-cap | Allowed for candidate expressions if coverage is sufficient. |
| `ema_bfq_10` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_bfq_20` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_bfq_250` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_bfq_30` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_bfq_5` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_bfq_60` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_bfq_90` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_hfq_10` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_hfq_20` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_hfq_250` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_hfq_30` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_hfq_5` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_hfq_60` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_hfq_90` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_qfq_10` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_qfq_20` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_qfq_250` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_qfq_30` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_qfq_5` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_qfq_60` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ema_qfq_90` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `emv_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `emv_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `emv_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `expma_12_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `expma_12_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `expma_12_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `expma_50_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `expma_50_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `expma_50_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `float_share` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `free_share` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `high` | price | Allowed for candidate expressions if coverage is sufficient. |
| `high_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `high_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `kdj_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `kdj_d_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `kdj_d_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `kdj_d_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `kdj_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `kdj_k_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `kdj_k_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `kdj_k_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `kdj_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ktn_down_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ktn_down_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ktn_down_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ktn_mid_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ktn_mid_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ktn_mid_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ktn_upper_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ktn_upper_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ktn_upper_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `low` | price | Allowed for candidate expressions if coverage is sufficient. |
| `low_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `low_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `lowdays` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ma_bfq_10` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_bfq_20` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_bfq_250` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_bfq_30` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_bfq_5` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_bfq_60` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_bfq_90` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_hfq_10` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_hfq_20` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_hfq_250` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_hfq_30` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_hfq_5` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_hfq_60` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_hfq_90` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_mass_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_mass_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_mass_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_qfq_10` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_qfq_20` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_qfq_250` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_qfq_30` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_qfq_5` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_qfq_60` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `ma_qfq_90` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `macd_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `macd_dea_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `macd_dea_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `macd_dea_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `macd_dif_bfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `macd_dif_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `macd_dif_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `macd_hfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `macd_qfq` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `madpo_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `madpo_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `madpo_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `maemv_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `maemv_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `maemv_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `maroc_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `maroc_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `maroc_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mass_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mass_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mass_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mfi_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mfi_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mfi_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mtm_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mtm_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mtm_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mtmma_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mtmma_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `mtmma_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `obv_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `obv_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `obv_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `open` | price | Allowed for candidate expressions if coverage is sufficient. |
| `open_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `open_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `pb` | valuation/market-cap | Allowed for candidate expressions if coverage is sufficient. |
| `pct_chg` | price | Allowed for candidate expressions if coverage is sufficient. |
| `pe` | valuation/market-cap | Allowed for candidate expressions if coverage is sufficient. |
| `pe_ttm` | valuation/market-cap | Allowed for candidate expressions if coverage is sufficient. |
| `pre_close` | price | Allowed for candidate expressions if coverage is sufficient. |
| `ps` | valuation/market-cap | Allowed for candidate expressions if coverage is sufficient. |
| `ps_ttm` | valuation/market-cap | Allowed for candidate expressions if coverage is sufficient. |
| `psy_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `psy_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `psy_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `psyma_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `psyma_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `psyma_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roc_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roc_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `roc_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `rsi_bfq_12` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `rsi_bfq_24` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `rsi_bfq_6` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `rsi_hfq_12` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `rsi_hfq_24` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `rsi_hfq_6` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `rsi_qfq_12` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `rsi_qfq_24` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `rsi_qfq_6` | technical | Allowed for candidate expressions if coverage is sufficient. |
| `taq_down_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `taq_down_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `taq_down_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `taq_mid_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `taq_mid_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `taq_mid_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `taq_up_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `taq_up_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `taq_up_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `topdays` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `total_mv` | valuation/market-cap | Allowed for candidate expressions if coverage is sufficient. |
| `total_share` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `trade_date` | identifier/date | Join key; do not use as a factor signal. |
| `trix_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `trix_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `trix_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `trma_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `trma_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `trma_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |
| `turnover_rate` | volume/liquidity | Allowed for candidate expressions if coverage is sufficient. |
| `turnover_rate_f` | volume/liquidity | Allowed for candidate expressions if coverage is sufficient. |
| `updays` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `vol` | volume/liquidity | Allowed for candidate expressions if coverage is sufficient. |
| `volume_ratio` | volume/liquidity | Allowed for candidate expressions if coverage is sufficient. |
| `vr_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `vr_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `vr_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `wr1_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `wr1_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `wr1_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `wr_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `wr_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `wr_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td1_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td1_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td1_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td2_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td2_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td2_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td3_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td3_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td3_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td4_bfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td4_hfq` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `xsii_td4_qfq` | raw | Allowed for candidate expressions if coverage is sufficient. |

### `stock_info_table`

- Update script: `数据库/daily_task/update_stock_info_table.py`
- Field count: 17

| Field | Role | Agent Usage |
| --- | --- | --- |
| `act_ent_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `act_name` | metadata | Use for filtering or labeling, not numeric DSL expressions. |
| `area` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `cnspell` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `curr_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `delist_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `enname` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `exchange` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `fullname` | metadata | Use for filtering or labeling, not numeric DSL expressions. |
| `industry` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `is_hs` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `list_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `list_status` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `market` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `name` | metadata | Use for filtering or labeling, not numeric DSL expressions. |
| `symbol` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |

### `stock_trading_suspension_info`

- Update script: `数据库/daily_task/update_stock_trading_suspension_info.py`
- Field count: 5

| Field | Role | Agent Usage |
| --- | --- | --- |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `suspend_timing` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `suspend_type` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `trade_date` | identifier/date | Join key; do not use as a factor signal. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |

### `stock_used_name`

- Update script: `数据库/daily_task/update_stock_used_name.py`
- Field count: 6

| Field | Role | Agent Usage |
| --- | --- | --- |
| `ann_date` | identifier/date | Join key; do not use as a factor signal. |
| `change_reason` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `end_date` | identifier/date | Join key; do not use as a factor signal. |
| `name` | metadata | Use for filtering or labeling, not numeric DSL expressions. |
| `start_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `ts_code` | identifier/date | Join key; do not use as a factor signal. |

### `sw_ind_code`

- Update script: `数据库/daily_task/update_sw_ind_code.py`
- Field count: 7

| Field | Role | Agent Usage |
| --- | --- | --- |
| `index_code` | identifier/date | Join key; do not use as a factor signal. |
| `industry_code` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `industry_name` | metadata | Use for filtering or labeling, not numeric DSL expressions. |
| `is_pub` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `level` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `parent_code` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `src` | raw | Allowed for candidate expressions if coverage is sufficient. |

### `trade_calendar`

- Update script: `数据库/daily_task/update_trade_calendar.py`
- Field count: 4

| Field | Role | Agent Usage |
| --- | --- | --- |
| `cal_date` | identifier/date | Join key; do not use as a factor signal. |
| `exchange` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `is_open` | raw | Allowed for candidate expressions if coverage is sufficient. |
| `pretrade_date` | raw | Allowed for candidate expressions if coverage is sufficient. |
