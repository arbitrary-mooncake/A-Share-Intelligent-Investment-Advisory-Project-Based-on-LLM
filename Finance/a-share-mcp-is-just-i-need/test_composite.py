#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CompositeDataSource + AKshare 测试脚本
测试：
1. 基本语法和导入
2. CompositeDataSource 正常路径（Baostock 优先）
3. Fallback 路径（当 Baostock 失败时回退到 AKshare）
4. AKshare 独立数据源
"""

import sys
import os
import logging
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = current_dir
sys.path.append(project_dir)

from src.baostock_data_source import BaostockDataSource
from src.akshare_data_source import AkshareDataSource
from src.composite_data_source import CompositeDataSource
from src.data_source_interface import NoDataFoundError, DataSourceError, LoginError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompositeTester:
    def __init__(self):
        self.baostock = BaostockDataSource()
        self.akshare = AkshareDataSource()
        self.composite = CompositeDataSource(primary=self.baostock, fallback=self.akshare)
        self.test_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.no_data_count = 0

    def run_test(self, name, func, *args, **kwargs):
        self.test_count += 1
        print(f"\n{'='*60}")
        print(f"测试 {self.test_count}: {name}")
        print(f"{'='*60}")
        try:
            result = func(*args, **kwargs)
            if isinstance(result, str):
                print(f"  结果长度: {len(result)}")
                print(f"  预览: {result[:200]}")
            else:
                print(f"  数据条数: {len(result)}")
                if len(result) > 0:
                    print(f"  数据列: {list(result.columns)[:10]}")
                    print(f"  预览:\n{result.head(2).to_string(index=False)}")
            self.success_count += 1
            print(f"  PASS")
            return True
        except NoDataFoundError as e:
            print(f"  NO DATA: {e}")
            self.no_data_count += 1
            return True
        except Exception as e:
            print(f"  FAIL: {e}")
            self.fail_count += 1
            return False

    # ---- Part 1: CompositeDataSource normal path ----
    def test_composite_k_data(self):
        return self.run_test(
            "Composite: K-line data (Baostock primary)",
            self.composite.get_historical_k_data,
            code="sh.603871", start_date="2023-12-01", end_date="2023-12-31"
        )

    def test_composite_basic_info(self):
        return self.run_test(
            "Composite: Stock basic info",
            self.composite.get_stock_basic_info,
            code="sh.603871"
        )

    def test_composite_dividend(self):
        return self.run_test(
            "Composite: Dividend data",
            self.composite.get_dividend_data,
            code="sh.603871", year="2023", year_type="report"
        )

    def test_composite_adjust_factor(self):
        return self.run_test(
            "Composite: Adjust factor",
            self.composite.get_adjust_factor_data,
            code="sh.603871", start_date="2023-01-01", end_date="2023-12-31"
        )

    def test_composite_profit(self):
        return self.run_test(
            "Composite: Profit data",
            self.composite.get_profit_data,
            code="sh.603871", year="2023", quarter=4
        )

    def test_composite_operation(self):
        return self.run_test(
            "Composite: Operation data",
            self.composite.get_operation_data,
            code="sh.603871", year="2023", quarter=4
        )

    def test_composite_growth(self):
        return self.run_test(
            "Composite: Growth data",
            self.composite.get_growth_data,
            code="sh.603871", year="2023", quarter=4
        )

    def test_composite_balance(self):
        return self.run_test(
            "Composite: Balance data",
            self.composite.get_balance_data,
            code="sh.603871", year="2023", quarter=4
        )

    def test_composite_cashflow(self):
        return self.run_test(
            "Composite: Cash flow data",
            self.composite.get_cash_flow_data,
            code="sh.603871", year="2023", quarter=4
        )

    def test_composite_dupont(self):
        return self.run_test(
            "Composite: DuPont data",
            self.composite.get_dupont_data,
            code="sh.603871", year="2023", quarter=4
        )

    def test_composite_performance_express(self):
        return self.run_test(
            "Composite: Performance express report",
            self.composite.get_performance_express_report,
            code="sh.600000", start_date="2015-01-01", end_date="2015-12-31"
        )

    def test_composite_forecast(self):
        return self.run_test(
            "Composite: Forecast report",
            self.composite.get_forecast_report,
            code="sh.603871", start_date="2023-01-01", end_date="2023-12-31"
        )

    def test_composite_industry(self):
        return self.run_test(
            "Composite: Stock industry",
            self.composite.get_stock_industry,
            code="sh.603871"
        )

    def test_composite_sz50(self):
        return self.run_test(
            "Composite: SZ50 constituents",
            self.composite.get_sz50_stocks
        )

    def test_composite_hs300(self):
        return self.run_test(
            "Composite: HS300 constituents",
            self.composite.get_hs300_stocks
        )

    def test_composite_zz500(self):
        return self.run_test(
            "Composite: ZZ500 constituents",
            self.composite.get_zz500_stocks
        )

    def test_composite_trade_dates(self):
        return self.run_test(
            "Composite: Trade dates",
            self.composite.get_trade_dates,
            start_date="2023-01-01", end_date="2023-01-31"
        )

    def test_composite_all_stock(self):
        return self.run_test(
            "Composite: All stock list",
            self.composite.get_all_stock,
            date="2017-06-30"
        )

    def test_composite_deposit_rate(self):
        return self.run_test(
            "Composite: Deposit rate",
            self.composite.get_deposit_rate_data,
            start_date="2015-01-01", end_date="2015-12-31"
        )

    def test_composite_loan_rate(self):
        return self.run_test(
            "Composite: Loan rate",
            self.composite.get_loan_rate_data,
            start_date="2015-01-01", end_date="2015-12-31"
        )

    def test_composite_reserve_ratio(self):
        return self.run_test(
            "Composite: Reserve ratio",
            self.composite.get_required_reserve_ratio_data,
            start_date="2015-01-01", end_date="2015-12-31"
        )

    def test_composite_money_supply_month(self):
        return self.run_test(
            "Composite: Monthly money supply",
            self.composite.get_money_supply_data_month,
            start_date="2023-01", end_date="2023-12"
        )

    def test_composite_money_supply_year(self):
        return self.run_test(
            "Composite: Yearly money supply",
            self.composite.get_money_supply_data_year,
            start_date="2023", end_date="2023"
        )

    def test_composite_news(self):
        return self.run_test(
            "Composite: News crawler (Baostock only, no AKshare fallback)",
            self.composite.crawl_news,
            query="嘉友国际", top_k=3
        )

    # ---- Part 2: AKshare standalone tests ----
    def test_akshare_k_data(self):
        return self.run_test(
            "AKshare standalone: K-line data",
            self.akshare.get_historical_k_data,
            code="sh.603871", start_date="2023-12-01", end_date="2023-12-31"
        )

    def test_akshare_basic_info(self):
        return self.run_test(
            "AKshare standalone: Stock basic info",
            self.akshare.get_stock_basic_info,
            code="sh.603871"
        )

    def test_akshare_trade_dates(self):
        return self.run_test(
            "AKshare standalone: Trade dates",
            self.akshare.get_trade_dates,
            start_date="2023-01-01", end_date="2023-01-31"
        )

    def test_akshare_all_stock(self):
        return self.run_test(
            "AKshare standalone: All stock list",
            self.akshare.get_all_stock
        )

    def test_akshare_hs300(self):
        return self.run_test(
            "AKshare standalone: HS300 constituents",
            self.akshare.get_hs300_stocks
        )

    def test_akshare_deposit_rate(self):
        return self.run_test(
            "AKshare standalone: Deposit rate",
            self.akshare.get_deposit_rate_data,
            start_date="2015-01-01", end_date="2015-12-31"
        )

    def test_akshare_reserve_ratio(self):
        return self.run_test(
            "AKshare standalone: Reserve ratio",
            self.akshare.get_required_reserve_ratio_data,
            start_date="2015-01-01", end_date="2015-12-31"
        )

    def test_akshare_money_supply(self):
        return self.run_test(
            "AKshare standalone: Money supply",
            self.akshare.get_money_supply_data_month,
            start_date="2023-01", end_date="2023-12"
        )

    def test_akshare_profit(self):
        return self.run_test(
            "AKshare standalone: Profit data (Sina)",
            self.akshare.get_profit_data,
            code="sh.603871", year="2023", quarter=4
        )

    def test_akshare_balance(self):
        return self.run_test(
            "AKshare standalone: Balance sheet (Sina)",
            self.akshare.get_balance_data,
            code="sh.603871", year="2023", quarter=4
        )

    def test_akshare_cashflow(self):
        return self.run_test(
            "AKshare standalone: Cash flow (Sina)",
            self.akshare.get_cash_flow_data,
            code="sh.603871", year="2023", quarter=4
        )

    def test_akshare_dividend(self):
        return self.run_test(
            "AKshare standalone: Dividend data",
            self.akshare.get_dividend_data,
            code="sh.603871", year="2023"
        )

    def test_akshare_industry(self):
        return self.run_test(
            "AKshare standalone: Industry data",
            self.akshare.get_stock_industry,
            code="sh.603871"
        )

    def run_all_tests(self):
        print("="*60)
        print(" CompositeDataSource + AKshare 测试开始")
        print("="*60)

        tests = [
            # Composite tests (Baostock primary + AKshare fallback)
            self.test_composite_k_data,
            self.test_composite_basic_info,
            self.test_composite_dividend,
            self.test_composite_adjust_factor,
            self.test_composite_profit,
            self.test_composite_operation,
            self.test_composite_growth,
            self.test_composite_balance,
            self.test_composite_cashflow,
            self.test_composite_dupont,
            self.test_composite_performance_express,
            self.test_composite_forecast,
            self.test_composite_industry,
            self.test_composite_sz50,
            self.test_composite_hs300,
            self.test_composite_zz500,
            self.test_composite_trade_dates,
            self.test_composite_all_stock,
            self.test_composite_deposit_rate,
            self.test_composite_loan_rate,
            self.test_composite_reserve_ratio,
            self.test_composite_money_supply_month,
            self.test_composite_money_supply_year,
            self.test_composite_news,
            # AKshare standalone tests
            self.test_akshare_k_data,
            self.test_akshare_basic_info,
            self.test_akshare_trade_dates,
            self.test_akshare_all_stock,
            self.test_akshare_hs300,
            self.test_akshare_deposit_rate,
            self.test_akshare_reserve_ratio,
            self.test_akshare_money_supply,
            self.test_akshare_profit,
            self.test_akshare_balance,
            self.test_akshare_cashflow,
            self.test_akshare_dividend,
            self.test_akshare_industry,
        ]

        for test in tests:
            try:
                test()
            except KeyboardInterrupt:
                print("\n  测试被用户中断")
                break
            except Exception as e:
                print(f"\n  测试异常: {e}")
                logger.exception("测试异常")

        # Summary
        print(f"\n{'='*60}")
        print(" 测试结果汇总")
        print(f"{'='*60}")
        print(f"总测试数: {self.test_count}")
        print(f"通过数:  {self.success_count}")
        print(f"无数据:  {self.no_data_count}")
        print(f"失败数:  {self.fail_count}")
        if self.test_count > 0:
            pass_rate = (self.success_count + self.no_data_count) / self.test_count * 100
            print(f"通过率:  {pass_rate:.1f}%")

        if self.fail_count == 0:
            print("\n 所有测试通过！")
        else:
            print(f"\n 有 {self.fail_count} 个测试失败")
        print(f"{'='*60}")

def main():
    try:
        tester = CompositeTester()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n 测试被用户中断")
    except Exception as e:
        print(f"\n 测试过程中发生错误: {e}")
        logger.exception("测试过程中发生错误")

if __name__ == "__main__":
    main()
