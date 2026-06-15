import { useState, type FormEvent } from "react"
import { useRouter } from "next/router"
import Head from "next/head"
import { BrandMark } from "@/components/brand-mark"
import { Button } from "@/components/ui/button"
import { authApi } from "@/lib/api/auth"
import { getApiErrorMessage } from "@/lib/http"
import { acceptInviteSchema, getValidationMessage } from "@/lib/validation/auth"
import {
  CheckCircle2,
  Eye,
  EyeOff,
  Loader2,
  Lock,
  Mail,
  ShieldCheck,
} from "lucide-react"

type PageState = "form" | "submitting" | "success" | "invalid-link"

export default function AcceptInvitePage() {
  const router = useRouter()

  // email comes from the invite link query param: /accept-invite?email=...
  const emailFromQuery = typeof router.query.email === "string"
    ? decodeURIComponent(router.query.email).trim()
    : ""

  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [pageState, setPageState] = useState<PageState>("form")
  const [formError, setFormError] = useState("")
  const [apiError, setApiError] = useState("")

  const visibleError = formError || apiError

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setFormError("")
    setApiError("")

    if (!emailFromQuery) {
      setPageState("invalid-link")
      return
    }

    const result = acceptInviteSchema.safeParse({
      email: emailFromQuery,
      password,
      confirmPassword,
    })

    if (!result.success) {
      setFormError(getValidationMessage(result.error))
      return
    }

    setPageState("submitting")

    try {
      await authApi.acceptInvite({
        email: result.data.email,
        password: result.data.password,
      })
      setPageState("success")
    } catch (err) {
      setApiError(getApiErrorMessage(err))
      setPageState("form")
    }
  }

  // Invalid / missing email in URL
  if (pageState === "invalid-link" || (router.isReady && !emailFromQuery)) {
    return (
      <>
        <Head>
          <title>Invalid Invite Link — Bangladesh Legal Advisor</title>
        </Head>
        <main className="grid min-h-screen place-items-center bg-background px-6">
          <div className="w-full max-w-sm text-center">
            <BrandMark size={56} className="mx-auto" />
            <h1 className="mt-5 font-heading text-xl font-bold">Invalid invite link</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              This link is missing or malformed. Please ask your administrator to
              resend the invitation.
            </p>
            <Button
              type="button"
              variant="outline"
              className="mt-6 w-full rounded-xl"
              onClick={() => router.replace("/login")}
            >
              Back to login
            </Button>
          </div>
        </main>
      </>
    )
  }

  // Success state
  if (pageState === "success") {
    return (
      <>
        <Head>
          <title>Account Activated — Bangladesh Legal Advisor</title>
        </Head>
        <main className="grid min-h-screen place-items-center bg-background px-6">
          <div className="w-full max-w-sm text-center">
            <span className="inline-flex size-16 items-center justify-center rounded-full bg-accent text-primary">
              <CheckCircle2 className="size-8" />
            </span>
            <h1 className="mt-5 font-heading text-xl font-bold">Account activated</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Your Lawyer account is ready. You can now sign in with your email and
              the password you just set.
            </p>
            <Button
              type="button"
              className="mt-6 h-11 w-full rounded-xl text-base"
              onClick={() => router.replace("/login")}
            >
              Go to login
            </Button>
          </div>
        </main>
      </>
    )
  }

  const isSubmitting = pageState === "submitting"

  return (
    <>
      <Head>
        <title>Set Your Password — Bangladesh Legal Advisor</title>
      </Head>
      <main className="grid min-h-screen grid-cols-1 lg:grid-cols-[1.1fr_1fr]">
        {/* Left panel */}
        <section className="relative hidden flex-col justify-between overflow-hidden bg-primary p-10 text-primary-foreground lg:flex">
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.12]"
            style={{
              backgroundImage: "radial-gradient(circle at 1px 1px, white 1px, transparent 0)",
              backgroundSize: "22px 22px",
            }}
            aria-hidden="true"
          />
          <div className="relative flex items-center gap-3">
            <BrandMark size={48} />
            <div className="leading-tight">
              <p className="text-sm/5 text-primary-foreground/80">Bangladesh Legal Advisor</p>
              <p className="font-heading text-lg font-bold">Bangladesh Legal Advisor</p>
            </div>
          </div>

          <div className="relative max-w-md">
            <span className="inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1 text-xs font-medium">
              <ShieldCheck className="size-3.5" />
              Lawyer invitation
            </span>
            <h1 className="mt-5 text-balance font-heading text-4xl font-bold leading-tight">
              You have been invited as a Lawyer
            </h1>
            <p className="mt-4 text-pretty leading-relaxed text-primary-foreground/85">
              Set a password to activate your account and start using the Bangladesh Legal Advisor platform.
            </p>
          </div>

          <p className="relative text-xs text-primary-foreground/70">
            © {new Date().getFullYear()} Bangladesh Legal Advisor. All rights reserved.
          </p>
        </section>

        {/* Right panel */}
        <section className="flex min-h-screen items-center justify-center overflow-y-auto bg-background px-6 py-8">
          <div className="w-full max-w-sm">
            {/* Mobile brand */}
            <div className="mb-8 flex flex-col items-center text-center lg:hidden">
              <BrandMark size={56} />
              <p className="mt-3 text-sm text-muted-foreground">Bangladesh Legal Advisor</p>
              <h1 className="font-heading text-2xl font-bold">Bangladesh Legal Advisor</h1>
            </div>

            <div className="rounded-2xl border border-border bg-card p-7 shadow-sm">
              <h2 className="font-heading text-xl font-bold text-card-foreground">
                Set your password
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">
                Create a password to activate your Lawyer account.
              </p>

              {/* Email display — read-only, taken from invite link */}
              <div className="mt-4 flex items-center gap-2.5 rounded-xl border border-border bg-muted/50 px-3.5 py-2.5">
                <Mail className="size-4 shrink-0 text-muted-foreground" />
                <span className="truncate text-sm font-medium text-foreground">
                  {emailFromQuery}
                </span>
              </div>

              {visibleError ? (
                <p className="mt-4 rounded-xl border border-destructive/25 bg-destructive/10 px-3.5 py-2.5 text-sm text-destructive">
                  {visibleError}
                </p>
              ) : null}

              <form className="mt-5 space-y-4" onSubmit={handleSubmit}>
                {/* Password */}
                <div className="space-y-1.5">
                  <label htmlFor="password" className="text-sm font-medium text-foreground">
                    Password
                  </label>
                  <div className="relative">
                    <Lock className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                    <input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={isSubmitting}
                      placeholder="Enter your password"
                      autoComplete="new-password"
                      className="h-11 w-full rounded-xl border border-input bg-background px-10 text-sm outline-none transition disabled:cursor-not-allowed disabled:opacity-60 focus:border-ring focus:ring-2 focus:ring-ring/30"
                    />
                    <button
                      type="button"
                      tabIndex={-1}
                      onClick={() => setShowPassword((p) => !p)}
                      className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground transition hover:text-foreground"
                      aria-label={showPassword ? "Hide password" : "Show password"}
                    >
                      {showPassword ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                    </button>
                  </div>
                </div>

                {/* Confirm password */}
                <div className="space-y-1.5">
                  <label htmlFor="confirm-password" className="text-sm font-medium text-foreground">
                    Confirm Password
                  </label>
                  <div className="relative">
                    <Lock className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                    <input
                      id="confirm-password"
                      type={showConfirm ? "text" : "password"}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      disabled={isSubmitting}
                      placeholder="Repeat your password"
                      autoComplete="new-password"
                      className="h-11 w-full rounded-xl border border-input bg-background px-10 text-sm outline-none transition disabled:cursor-not-allowed disabled:opacity-60 focus:border-ring focus:ring-2 focus:ring-ring/30"
                    />
                    <button
                      type="button"
                      tabIndex={-1}
                      onClick={() => setShowConfirm((p) => !p)}
                      className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground transition hover:text-foreground"
                      aria-label={showConfirm ? "Hide password" : "Show password"}
                    >
                      {showConfirm ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                    </button>
                  </div>
                  {/* Match indicator — only frontend UX, not a validation rule */}
                  {confirmPassword.length > 0 ? (
                    <p className={`text-xs ${password === confirmPassword ? "text-primary" : "text-destructive"}`}>
                      {password === confirmPassword ? "✓ Passwords match" : "✗ Passwords do not match"}
                    </p>
                  ) : null}
                </div>

                <Button
                  type="submit"
                  size="lg"
                  disabled={isSubmitting}
                  className="h-11 w-full rounded-xl text-base"
                >
                  <span className="inline-flex size-4 items-center justify-center">
                    {isSubmitting ? <Loader2 className="size-4 animate-spin" /> : null}
                  </span>
                  Activate account
                </Button>
              </form>
            </div>
          </div>
        </section>
      </main>
    </>
  )
}
