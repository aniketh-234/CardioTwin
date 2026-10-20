"""Run against python -m backend.server. Requires playwright + Chromium."""
import json,re
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8000');expect(page.locator('#patient-name')).to_have_text(re.compile('SIM-'))
    page.locator('#reset').click();expect(page.locator('#updates')).to_have_text('0')
    page.locator('#scenario-run').click();expect(page.locator('#scenario-result')).to_have_text(re.compile('hypothetical'))
    page.locator('#ingest').click();expect(page.locator('#updates')).to_have_text('1')
    page.locator('#dropout').click();expect(page.locator('#risk')).to_have_text('Withheld')
    page.locator('#reset').click();expect(page.locator('#updates')).to_have_text('0')
    page.locator('[data-view=evidence]').click();page.wait_for_selector('#metrics-body tr')
    assert page.locator('#metrics-body tr').count()==8
    expect(page.locator('#hybrid-metrics-row')).to_contain_text('Hybrid CNN + rules')
    expect(page.locator('#hybrid-metrics-row')).to_contain_text('79.3%')
    expect(page.locator('#hybrid-metrics-row')).to_contain_text('50.3%')
    page.locator('[data-view=journal]').click();page.wait_for_selector('.audit-row')
    page.locator('[data-view=twin]').click()
    Path('../qa').mkdir(exist_ok=True)
    page.screenshot(path='../qa/dashboard-desktop.png',full_page=True)
    page.set_viewport_size({'width':390,'height':844});page.screenshot(path='../qa/dashboard-mobile.png',full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert not errors,errors
    browser.close()
    print('PASS: browser startup, scenario, ingestion, dropout abstention, reset, evidence, audit, mobile layout; no JS errors.')
