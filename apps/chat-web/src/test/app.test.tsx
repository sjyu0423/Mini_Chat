import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App, { DISCLAIMER } from "../App";

const READY = {
  status: "ready",
  model: {
    name: "m5_sft_chat",
    parameters: 13989888,
    context_length: 512,
    n_layer: 6,
    n_head: 6,
    n_embd: 384,
    vocab_size: 8192,
    tokenizer_fingerprint: "e6be77cc",
    device: "mps",
    load_seconds: 0.4,
  },
  defaults: { temperature: 0.7, top_k: 40, repetition_penalty: 1.2, max_new_tokens: 200 },
  limits: {
    max_message_chars: 2000,
    max_new_tokens: 256,
    min_new_tokens: 1,
    min_temperature: 0.1,
    max_temperature: 1.5,
  },
};

function reply(text: string, stop = "end_of_turn", tokens = 7) {
  return {
    reply: text,
    stop_reason: stop,
    new_tokens: tokens,
    prompt_tokens: 12,
    dropped_history_messages: 0,
    duration_ms: 320,
    tokens_per_s: 21.9,
    seed: 5,
    settings: { temperature: 0.7, top_k: 40, repetition_penalty: 1.2, max_new_tokens: 200 },
  };
}

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

type Handler = (body: { message: string; history: unknown[]; settings: unknown }) => Promise<Response>;

function mockServer(chat: Handler, health: () => Promise<Response> = () => json(READY)) {
  const calls: { message: string; history: unknown[]; settings: unknown }[] = [];
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    if (url === "/api/health") return health();
    const body = JSON.parse(String(init?.body));
    calls.push(body);
    return chat(body);
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function ready() {
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Message" })).toBeEnabled());
  return screen.getByRole("textbox", { name: "Message" });
}

describe("App", () => {
  it("shows the required disclaimer and the about panel", async () => {
    mockServer(() => json(reply("hi")));
    render(<App />);
    expect(screen.getByText(DISCLAIMER)).toBeInTheDocument();
    await ready();
    await userEvent.click(screen.getByRole("button", { name: "About" }));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(/14 million parameters/)).toBeInTheDocument();
    expect(within(dialog).getByText(/No external model API/)).toBeInTheDocument();
    expect(within(dialog).getByText(/13,989,888/)).toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "About" }));
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("disables input while the model is loading", async () => {
    let healthCalls = 0;
    mockServer(
      () => json(reply("x")),
      () => json(healthCalls++ === 0 ? { status: "loading" } : READY),
    );
    render(<App healthPollMs={10} />);
    await waitFor(() => expect(screen.getByText("Loading the model…")).toBeInTheDocument());
    expect(screen.getByRole("textbox", { name: "Message" })).toBeDisabled();
    await ready();
    expect(screen.queryByText("Loading the model…")).not.toBeInTheDocument();
  });

  it("renders user and assistant messages with stop reason, tokens and duration", async () => {
    mockServer(() => json(reply("Python is a programming language.")));
    render(<App />);
    const box = await ready();
    await userEvent.type(box, "What is Python?{Enter}");
    expect(await screen.findByText("Python is a programming language.")).toBeInTheDocument();
    expect(screen.getByTestId("msg-user")).toHaveTextContent("What is Python?");
    const assistant = screen.getByTestId("msg-assistant");
    expect(assistant).toHaveTextContent("ended naturally");
    expect(assistant).toHaveTextContent("7 tokens");
    expect(assistant).toHaveTextContent("320 ms");
    expect(box).toHaveValue("");
  });

  it("marks truncated replies as cut off", async () => {
    mockServer(() => json(reply("loop loop loop", "max_tokens", 200)));
    render(<App />);
    await userEvent.type(await ready(), "Hi{Enter}");
    expect(await screen.findByText("cut off at token limit")).toBeInTheDocument();
  });

  it("Shift+Enter inserts a newline instead of sending", async () => {
    const calls = mockServer(() => json(reply("ok")));
    render(<App />);
    const box = await ready();
    await userEvent.type(box, "line one{Shift>}{Enter}{/Shift}line two");
    expect(box).toHaveValue("line one\nline two");
    expect(calls).toHaveLength(0);
    await userEvent.type(box, "{Enter}");
    await screen.findByText("ok");
    expect(calls[0]?.message).toBe("line one\nline two");
  });

  it("shows a loading state while generating and sends history on the next turn", async () => {
    let release: (r: Response) => void = () => {};
    const calls = mockServer(
      (b) =>
        b.message === "first"
          ? new Promise<Response>((res) => (release = res))
          : json(reply("second answer")),
    );
    render(<App />);
    const box = await ready();
    await userEvent.type(box, "first{Enter}");
    expect(screen.getByTestId("pending")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    await act(async () => release(new Response(JSON.stringify(reply("first answer")))));
    await screen.findByText("first answer");
    expect(screen.queryByTestId("pending")).not.toBeInTheDocument();

    await userEvent.type(box, "second{Enter}");
    await screen.findByText("second answer");
    expect(calls[1]?.history).toEqual([
      { role: "user", content: "first" },
      { role: "assistant", content: "first answer" },
    ]);
    expect(calls[1]?.settings).toEqual({ temperature: 0.7, max_new_tokens: 200 });
  });

  it("shows server errors, keeps the draft and marks the message unanswered", async () => {
    mockServer(() => json({ detail: "message longer than 2000 characters" }, 422));
    render(<App />);
    await userEvent.type(await ready(), "bad request{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("message longer than 2000 characters");
    expect(screen.getByText("Not answered (request failed)")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("bad request");
  });

  it("reports an unreachable server", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    render(<App healthPollMs={10_000} />);
    expect(await screen.findByText(/Cannot reach the local model server/)).toBeInTheDocument();
  });

  it("New chat clears the conversation and session storage", async () => {
    mockServer(() => json(reply("an answer")));
    render(<App />);
    await userEvent.type(await ready(), "hello{Enter}");
    await screen.findByText("an answer");
    expect(sessionStorage.getItem("mini-gpt-chat:v1")).toContain("an answer");
    await userEvent.click(screen.getByRole("button", { name: "New chat" }));
    expect(screen.queryByText("an answer")).not.toBeInTheDocument();
    expect(screen.getByText("Ask the model something")).toBeInTheDocument();
    expect(sessionStorage.getItem("mini-gpt-chat:v1")).toBeNull();
  });

  it("restores the conversation from session storage on reload", async () => {
    mockServer(() => json(reply("kept")));
    const { unmount } = render(<App />);
    await userEvent.type(await ready(), "remember{Enter}");
    await screen.findByText("kept");
    unmount();
    render(<App />);
    expect(screen.getByText("kept")).toBeInTheDocument();
  });

  it("sends advanced settings when changed", async () => {
    const calls = mockServer(() => json(reply("ok")));
    render(<App />);
    const box = await ready();
    await userEvent.click(screen.getByRole("button", { name: "Settings" }));
    const sliders = [
      screen.getByRole("slider", { name: "Temperature" }),
      screen.getByRole("slider", { name: "Max new tokens" }),
    ];
    // fireEvent-free: set values through the input elements
    await act(async () => {
      for (const [el, v] of [
        [sliders[0], "1.2"],
        [sliders[1], "64"],
      ] as const) {
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
        setter.call(el, v);
        el!.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
    await userEvent.type(box, "Hi{Enter}");
    await screen.findByText("ok");
    expect(calls[0]?.settings).toEqual({ temperature: 1.2, max_new_tokens: 64 });
  });
});
