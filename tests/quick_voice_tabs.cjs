// Run with Playwright installed and VOXFORGE_TEST_URL pointing to a local Playground.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

test('Quick voice keeps its text and profile requirement across workspace tabs', async () => {
    const url = new URL(process.env.VOXFORGE_TEST_URL);
    assert.equal(url.hostname, '127.0.0.1');
    const browser = await chromium.launch({ channel: 'chrome' });
    try {
        const page = await browser.newPage();
        await page.goto(url.href);
        const text = page.getByRole('textbox', { name: 'Seslendirilecek metin' });
        const generate = page.getByRole('button', { name: 'Seslendir', exact: true });
        await text.fill('Kısa bir deneme.');
        assert.ok(await generate.isDisabled());
        for (const name of ['Projeler', '2. Sesler', '3. Üretim', 'Sahneli anlatım',
            'Kayıtlar', 'Kaliteyi denetle', 'Hızlı ses']) {
            await page.getByRole('tab', { name, exact: true }).click();
        }
        assert.equal(await text.inputValue(), 'Kısa bir deneme.');
        await page.getByText('Ses ve üretim ayarları', { exact: true }).click();
        await page.getByRole('radio', { name: 'Mock', exact: true }).check();
        await page.waitForFunction(() => !document.querySelector('.vf-go').disabled);
        assert.ok(await generate.isEnabled());
        await page.getByRole('radio', { name: 'VoxCPM2', exact: true }).first().check();
        await page.waitForFunction(() => document.querySelector('.vf-go').disabled);
    } finally {
        await browser.close();
    }
});
