import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import data from "../showcase/data.json";
import { fact, recorded } from "../showcase/content";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("showcase", () => {
  it("static demo never calls the API and shows recorded conversations by default", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<App staticDemo />);
    expect(screen.queryByRole("button", { name: "Live chat" })).not.toBeInTheDocument();
    expect(screen.getAllByTestId("recorded-item")).toHaveLength(recorded.length);
    expect(screen.getByText(/No model runs on this page/)).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Message" })).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("recorded replies are the real data, with the 30M comparison on demand", async () => {
    vi.stubGlobal("fetch", vi.fn());
    render(<App staticDemo />);
    const first = screen.getAllByTestId("recorded-item")[0]!;
    expect(within(first).getByText(recorded[0]!.replies.m5.reply)).toBeInTheDocument();
    expect(within(first).queryByText("Model · 30M (research)")).not.toBeInTheDocument();
    await userEvent.click(screen.getByLabelText("Compare with the 30M research model"));
    expect(within(first).getByText("Model · 30M (research)")).toBeInTheDocument();
  });

  it("project page shows the key finding with its uncertainty interval", async () => {
    vi.stubGlobal("fetch", vi.fn());
    render(<App staticDemo />);
    await userEvent.click(screen.getByRole("button", { name: "Project" }));
    expect(screen.getByText(fact("strict_success_ci"))).toBeInTheDocument();
    expect(screen.getByText(fact("m5_parameters"))).toBeInTheDocument();
    expect(screen.getByText(/did not produce\s+a statistically clear improvement/)).toBeInTheDocument();
  });

  it("every fact shown is backed by docs/showcase/facts.json", () => {
    expect(() => fact("no_such_fact")).toThrow();
    for (const id of ["m5_parameters", "m63_parameters", "strict_success", "blind_review"]) {
      expect(fact(id)).toBe((data.facts as Record<string, { value: string }>)[id]!.value);
    }
  });

  it("offers the recorded conversations when the live server is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    render(<App healthPollMs={10_000} />);
    await userEvent.click(await screen.findByRole("button", { name: "See recorded conversations instead" }));
    expect(screen.getAllByTestId("recorded-item").length).toBeGreaterThan(0);
  });
});
