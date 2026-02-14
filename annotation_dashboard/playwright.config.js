// Optional E2E setup. Requires Node + Playwright.
module.exports = {
  testDir: "./tests",
  timeout: 60_000,
  use: {
    baseURL: "http://localhost:8000",
    headless: true,
  },
};
