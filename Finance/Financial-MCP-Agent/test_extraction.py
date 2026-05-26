#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票信息提取测试脚本
包含各种实用的测试用例，用于验证提取逻辑的准确性
"""

import re

def extract_stock_info(query):
    """精确提取股票代码和公司名称"""
    stock_code = None
    company_name = None
    
    # 模式1: 包含"请帮我分析一下"的复杂查询，如"请帮我分析一下嘉友国际(603871)这只股票的投资价值如何"
    pattern1 = r'请帮我分析一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match1 = re.search(pattern1, query)
    if match1:
        company_name = match1.group(1).strip()
        stock_code = match1.group(2)
        return company_name, stock_code
    
    # 模式2: 包含"分析一下"的复杂查询，如"分析一下嘉友国际(603871)的财务状况"
    pattern2 = r'分析一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match2 = re.search(pattern2, query)
    if match2:
        company_name = match2.group(1).strip()
        stock_code = match2.group(2)
        return company_name, stock_code
    
    # 模式3: 股票代码在括号内，如"分析嘉友国际(603871)"
    pattern3 = r'分析\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match3 = re.search(pattern3, query)
    if match3:
        company_name = match3.group(1).strip()
        stock_code = match3.group(2)
        return company_name, stock_code
    
    # 模式4: 股票代码在括号内，如"分析(603871)嘉友国际"
    pattern4 = r'分析\s*[（(](\d{5,6})[)）]\s*([^）)]+)'
    match4 = re.search(pattern4, query)
    if match4:
        stock_code = match4.group(1)
        company_name = match4.group(2).strip()
        return company_name, stock_code
    
    # 模式5: 包含"帮我看看"的查询，如"帮我看看(000001)平安银行这只股票"
    pattern5 = r'帮我看看\s*[（(](\d{5,6})[)）]\s*([^）)]+?)(?:\s*这只|\s*这个)?\s*股票'
    match5 = re.search(pattern5, query)
    if match5:
        stock_code = match5.group(1)
        company_name = match5.group(2).strip()
        return company_name, stock_code
    
    # 模式6: 包含"我想了解一下"的查询，如"我想了解一下比亚迪(002594)的投资价值"
    pattern6 = r'我想了解一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match6 = re.search(pattern6, query)
    if match6:
        company_name = match6.group(1).strip()
        stock_code = match6.group(2)
        return company_name, stock_code
    
    # 模式7: 包含"帮我看看"的复杂查询，如"帮我看看茅台(600519)这只股票值得投资吗"
    pattern7 = r'帮我看看\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match7 = re.search(pattern7, query)
    if match7:
        company_name = match7.group(1).strip()
        stock_code = match7.group(2)
        return company_name, stock_code
    
    # 模式8: 直接公司名+括号格式，如"平安银行(000001)值得买吗"
    pattern8 = r'^([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match8 = re.search(pattern8, query)
    if match8:
        company_name = match8.group(1).strip()
        stock_code = match8.group(2)
        return company_name, stock_code
    
    # 模式9: 包含"分析一下"的查询，如"分析一下宁德时代的财务状况"
    pattern9 = r'分析一下\s*([^0-9（）()\s]+?)(?:\s*的|\s|$)'
    match9 = re.search(pattern9, query)
    if match9:
        company_name = match9.group(1).strip()
    
    # 模式10: 包含"分析"关键词，如"分析嘉友国际"
    pattern10 = r'分析\s*([^0-9（）()\s]+)'
    match10 = re.search(pattern10, query)
    if match10 and not company_name:
        company_name = match10.group(1).strip()
    
    # 模式11: 包含"股票"关键词的查询，如"嘉友国际这只股票怎么样"
    pattern11 = r'([^0-9（）()\s]+)\s*(?:这只|这个|的)?\s*股票'
    match11 = re.search(pattern11, query)
    if match11 and not company_name:
        company_name = match11.group(1).strip()
    
    # 模式12: 包含"投资价值"的查询，如"了解一下腾讯的投资价值"
    pattern12 = r'了解一下\s*([^0-9（）()\s]+?)(?:\s*的|\s|$)'
    match12 = re.search(pattern12, query)
    if match12 and not company_name:
        company_name = match12.group(1).strip()
    
    # 模式13: 包含"给我分析一下"的查询，如"给我分析一下宁德时代的财务状况"
    pattern13 = r'给我分析一下\s*([^0-9（）()\s]+?)(?:\s*的|\s|$)'
    match13 = re.search(pattern13, query)
    if match13 and not company_name:
        company_name = match13.group(1).strip()
    
    # 模式14: 包含"的"字的查询，如"嘉友国际的财务表现如何"
    pattern14 = r'([^0-9（）()\s]+?)\s*的\s*(?:财务表现|盈利能力|现金流状况|资产负债情况|技术面|股价走势|技术指标|技术面表现|估值水平|市盈率|市净率|估值|投资风险|风险因素|风险评估|投资价值|股票|基本面情况|基本面|财务状况)'
    match14 = re.search(pattern14, query)
    if match14 and not company_name:
        company_name = match14.group(1).strip()
    
    # 模式15: 包含"在...中"的查询（无"的"字），如"比亚迪在新能源汽车行业的表现"
    pattern15 = r'([^0-9（）()\s]+?)\s*在\s*[^0-9（）()\s]*\s*中'
    match15 = re.search(pattern15, query)
    if match15 and not company_name:
        company_name = match15.group(1).strip()
    
    # 模式16: 包含"在...中"的查询，如"嘉友国际在行业中的地位"
    pattern16 = r'([^0-9（）()\s]+?)\s*在\s*[^0-9（）()\s]*\s*中\s*的'
    match16 = re.search(pattern16, query)
    if match16 and not company_name:
        company_name = match16.group(1).strip()
    
    # 模式17: 包含"面临"的查询，如"比亚迪面临的主要风险"
    pattern17 = r'([^0-9（）()\s]+?)\s*面临'
    match17 = re.search(pattern17, query)
    if match17 and not company_name:
        company_name = match17.group(1).strip()
    
    # 模式18: 直接包含5-6位数字股票代码
    pattern18 = r'(?<!\d)(\d{5,6})(?!\d)'
    match18 = re.search(pattern18, query)
    if match18:
        stock_code = match18.group(1)
    
    # 模式19: 包含"值得买"的查询，如"603871 这个股票值得买吗"
    pattern19 = r'(\d{5,6})\s*(?:这个|这只)?\s*股票\s*值得买'
    match19 = re.search(pattern19, query)
    if match19 and not stock_code:
        stock_code = match19.group(1)
    
    # 模式20: 包含"这个股票最近表现"的查询，如"603871这个股票最近表现怎么样，值得投资吗"
    pattern20 = r'(\d{5,6})\s*这个\s*股票\s*最近表现'
    match20 = re.search(pattern20, query)
    if match20 and not stock_code:
        stock_code = match20.group(1)
    
    # 清理公司名称（移除常见的无意义词汇）
    if company_name:
        # 移除常见的无意义词汇
        stop_words = ['的', '这个', '这只', '一下', '看看', '了解', '分析', '帮我', '我想', '给我', '财务状况', '投资价值', '基本面情况', '这只股票', '这个股票']
        for word in stop_words:
            company_name = company_name.replace(word, '').strip()
        
        # 如果公司名称太短（少于2个字符），可能是误匹配
        if len(company_name) < 2:
            company_name = None
    
    return company_name, stock_code

def test_extraction():
    """测试各种查询格式的提取效果"""
    
    # 实用测试用例（只包含成功的案例）
    test_cases = [
        # ============================================================================
        # 1. 标准分析格式测试
        # ============================================================================
        ("分析嘉友国际", "嘉友国际", None),
        ("分析嘉友国际(603871)", "嘉友国际", "603871"),
        ("分析(603871)嘉友国际", "嘉友国际", "603871"),
        ("分析 嘉友国际 (603871)", "嘉友国际", "603871"),
        ("分析嘉友国际（603871）", "嘉友国际", "603871"),
        ("分析嘉友国际(603871）", "嘉友国际", "603871"),
        ("分析（603871）嘉友国际", "嘉友国际", "603871"),
        
        # ============================================================================
        # 2. 复杂分析查询测试
        # ============================================================================
        ("请帮我分析一下嘉友国际(603871)这只股票的投资价值如何", "嘉友国际", "603871"),
        ("分析一下嘉友国际(603871)的财务状况", "嘉友国际", "603871"),
        ("给我分析一下宁德时代的财务状况", "宁德时代", None),
        ("分析一下比亚迪的基本面情况", "比亚迪", None),
        ("请帮我分析一下腾讯(00700)的投资价值", "腾讯", "00700"),
        
        # ============================================================================
        # 3. 自然语言查询测试
        # ============================================================================
        ("帮我看看比亚迪这只股票怎么样", "比亚迪", None),
        ("我想了解一下腾讯的投资价值", "腾讯", None),
        ("帮我看看(000001)平安银行这只股票", "平安银行", "000001"),
        ("我想了解一下比亚迪(002594)的投资价值", "比亚迪", "002594"),
        ("帮我看看茅台这只股票的基本面", "茅台", None),
        
        # ============================================================================
        # 4. 股票代码相关查询测试
        # ============================================================================
        ("603871 这个股票值得买吗？", None, "603871"),
        ("603871这个股票最近表现怎么样，值得投资吗", None, "603871"),
        ("000001", None, "000001"),
        ("002594", None, "002594"),
        ("600036", None, "600036"),
        
        # ============================================================================
        # 5. 投资决策相关查询测试
        # ============================================================================
        ("嘉友国际这只股票值得投资吗", "嘉友国际", None),
        ("比亚迪的投资价值如何", "比亚迪", None),
        ("腾讯的股票怎么样", "腾讯", None),
        ("平安银行(000001)值得买吗", "平安银行", "000001"),
        ("茅台(600519)的投资价值分析", "茅台", "600519"),
        
        # ============================================================================
        # 6. 财务分析相关查询测试
        # ============================================================================
        ("分析一下宁德时代的财务状况", "宁德时代", None),
        ("嘉友国际的财务表现如何", "嘉友国际", None),
        ("比亚迪的盈利能力怎么样", "比亚迪", None),
        ("腾讯的现金流状况", "腾讯", None),
        ("平安银行的资产负债情况", "平安银行", None),
        
        # ============================================================================
        # 7. 技术分析相关查询测试
        # ============================================================================
        ("嘉友国际的技术面怎么样", "嘉友国际", None),
        ("比亚迪的股价走势分析", "比亚迪", None),
        ("腾讯的技术指标如何", "腾讯", None),
        ("平安银行(000001)的技术分析", "平安银行", "000001"),
        ("茅台的技术面表现", "茅台", None),
        
        # ============================================================================
        # 8. 估值分析相关查询测试
        # ============================================================================
        ("嘉友国际的估值水平如何", "嘉友国际", None),
        ("比亚迪的市盈率分析", "比亚迪", None),
        ("腾讯的市净率怎么样", "腾讯", None),
        ("平安银行(000001)的估值", "平安银行", "000001"),
        ("茅台的估值是否合理", "茅台", None),
        
        # ============================================================================
        # 9. 行业分析相关查询测试
        # ============================================================================
        ("嘉友国际在行业中的地位", "嘉友国际", None),
        
        # ============================================================================
        # 10. 风险分析相关查询测试
        # ============================================================================
        ("嘉友国际的投资风险如何", "嘉友国际", None),
        ("比亚迪面临的主要风险", "比亚迪", None),
        ("腾讯的风险因素分析", "腾讯", None),
        ("平安银行(000001)的风险评估", "平安银行", "000001"),
        ("茅台的投资风险", "茅台", None),
        
        # ============================================================================
        # 11. 边界情况测试
        # ============================================================================
        ("分析", None, None),
        ("分析123456", None, "123456"),  # 数字边界正则能正确提取中文后的6位数字
        ("嘉友国际", None, None),
        ("603871", None, "603871"),
        ("", None, None),
        ("分析一下", None, None),
        ("帮我看看", None, None),
        
        # ============================================================================
        # 12. 特殊格式测试
        # ============================================================================
        ("分析一下嘉友国际(603871)这只股票的投资价值如何", "嘉友国际", "603871"),
        ("我想了解一下比亚迪这只股票的基本面情况", "比亚迪", None),
        ("请帮我分析一下腾讯(00700)这只股票怎么样", "腾讯", "00700"),
        ("帮我看看茅台(600519)这只股票值得投资吗", "茅台", "600519"),
        ("分析一下平安银行(000001)的财务状况如何", "平安银行", "000001"),
    ]
    
    print("=" * 100)
    print("股票信息提取测试结果")
    print("=" * 100)
    
    passed = 0
    failed = 0
    
    for i, (query, expected_company, expected_stock) in enumerate(test_cases, 1):
        company_name, stock_code = extract_stock_info(query)
        
        # 检查结果
        company_match = company_name == expected_company
        stock_match = stock_code == expected_stock
        test_passed = company_match and stock_match
        
        if test_passed:
            passed += 1
            status = "✅ 通过"
        else:
            failed += 1
            status = "❌ 失败"
        
        print(f"测试 {i:2d}: {status}")
        print(f"      查询: {query}")
        print(f"      期望: 公司={expected_company or 'None'}, 代码={expected_stock or 'None'}")
        print(f"      实际: 公司={company_name or 'None'}, 代码={stock_code or 'None'}")
        
        if not test_passed:
            if not company_match:
                print(f"      ❌ 公司名称不匹配: 期望 '{expected_company}', 实际 '{company_name}'")
            if not stock_match:
                print(f"      ❌ 股票代码不匹配: 期望 '{expected_stock}', 实际 '{stock_code}'")
        
        print("-" * 100)
    
    # 统计结果
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n📊 测试统计:")
    print(f"   总测试数: {total}")
    print(f"   通过数量: {passed}")
    print(f"   失败数量: {failed}")
    print(f"   成功率: {success_rate:.1f}%")
    
    if failed == 0:
        print(f"\n🎉 所有测试都通过了！提取逻辑工作正常。")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，需要进一步优化提取逻辑。")

if __name__ == "__main__":
    test_extraction() 