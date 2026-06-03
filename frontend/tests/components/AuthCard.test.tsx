import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const mocks = vi.hoisted(() => ({
  getProviders: vi.fn(),
  login: vi.fn(),
  signup: vi.fn(),
  replace: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mocks.replace }),
  useSearchParams: () => new URLSearchParams(""),
}));

vi.mock("@/lib/api", async () => {
  const real = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...real,
    authApi: { ...real.authApi, getProviders: mocks.getProviders, oauthStartUrl: (p: string) => `http://b/${p}` },
  };
});

vi.mock("@/lib/auth/AuthContext", () => ({
  useAuth: () => ({ login: mocks.login, signup: mocks.signup }),
}));

import { AuthCard } from "@/components/auth/AuthCard";

beforeEach(() => {
  Object.values(mocks).forEach((m) => m.mockReset());
  mocks.getProviders.mockResolvedValue({ google: true, github: true, password: true, auth_required: true });
});

afterEach(() => vi.restoreAllMocks());

describe("AuthCard", () => {
  it("renders the login form with social buttons", async () => {
    render(<AuthCard mode="login" />);
    expect(screen.getByRole("heading", { name: /welcome back/i })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /continue with google/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue with github/i })).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
  });

  it("submits login credentials and redirects", async () => {
    mocks.login.mockResolvedValue(undefined);
    render(<AuthCard mode="login" />);
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "password123" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(mocks.login).toHaveBeenCalledWith("a@b.com", "password123"));
    await waitFor(() => expect(mocks.replace).toHaveBeenCalledWith("/"));
  });

  it("shows an error when login fails", async () => {
    const { ApiError } = await import("@/lib/api");
    mocks.login.mockRejectedValue(new ApiError({ status: 401, code: "invalid_credentials", message: "Invalid email or password." }));
    render(<AuthCard mode="login" />);
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrongpass1" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/invalid email or password/i);
  });

  it("signup mode shows the name field and a create button", async () => {
    render(<AuthCard mode="signup" />);
    expect(screen.getByRole("heading", { name: /create your account/i })).toBeInTheDocument();
    expect(screen.getByLabelText("Name")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create account" })).toBeInTheDocument();
  });
});
