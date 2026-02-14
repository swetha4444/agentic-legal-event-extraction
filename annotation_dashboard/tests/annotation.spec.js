const { test, expect } = require("@playwright/test");
const path = require("path");

const sampleCsv = path.resolve(__dirname, "../sample.csv");

test("annotate a small CSV and update progress", async ({ page }) => {
  await page.goto("/");

  await page.setInputFiles("#fileInput", sampleCsv);

  await expect(page.getByText("sample.csv loaded")).toBeVisible();
  await page.getByRole("button", { name: "Continue to annotation" }).click();

  await expect(page.getByText("1 / 3")).toBeVisible();
  await page.getByRole("button", { name: /Fact/ }).click();
  await page.getByRole("button", { name: /Non-Fact/ }).click();

  await expect(page.getByText("2 / 3")).toBeVisible();
});
