import { useState, type FormEvent } from "react"
import { useRouter } from "next/router"
import Head from "next/head"
import { BrandMark } from "@/components/brand-mark"
import { Button } from "@/components/ui/button"
import { usersApi } from "@/lib/api/users"
import { getApiErrorMessage } from "@/lib/http"
import { resetPasswordSchema, getValidationMessage } from "@/lib/validation/auth"
import {
  CheckCircle2,
  Eye,
  EyeOff,
  Loader2,
  Lock,
} from "lucide-react"

type PageState = "form" | "submitting" | "success" | "invalid-link"

export default function ResetPasswordPage() {
  const router = useRouter()

  // token comes from the reset link query param: /reset-password?token=...
  const tokenFromQuery = typeof router.query.token === "string"
    ? decodeURIComponent(router.query.token).trim()
    : ""

  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [pageState, setPageState] = useState<PageState>("form")
  const [formError, setFormError] = useState("")
  const [apiError, setApiError] = useState("")
  const [successMsg, setSuccessMsg] = useState("")

  const visibleError = formError || apiError

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setFormError("")
    setApiError("")

    if (!tokenFromQuery) {
      setPageState("invalid-link")
      return
    }

    const result = resetPasswordSchema.safeParse({
      token: tokenFromQuery,
      new_password: password,
      confirmPassword,
    })

    if (!result.success) {
      setFormError(getValidationMessage(result.error))
      return
    }

    setPageState("submitting")

    try {
      const backendMsg = await usersApi.resetPassword({
        token: result.data.token,
        new_password: result.data.new_password,
      })
      const msgText = typeof backendMsg === "string" ? backendMsg : (backendMsg as any)?.detail || "Your password has been reset successfully."
      setSuccessMsg(msgText)
      setPageState("success")
    } catch (err) {
      setApiError(getApiErrorMessage(err))
      setPageState("form")
    }
  }

  // Invalid / missing token in URL
  if (pageState === "invalid-link" || (router.isReady && !tokenFromQuery)) {
    return (
      <>
        <Head>
          <title>Invalid Reset Link — Bangladesh Legal Advisor</title>
        </Head>
        <main className="relative min-h-dvh overflow-hidden bg-gradient-to-b from-[#e0f2fe] via-[#bae6fd] to-white flex items-center justify-center px-6 py-8">
          <div className="w-full max-w-xl animate-in fade-in zoom-in-95 duration-500">
            <div className="mb-8 flex flex-col items-center text-center">
              <BrandMark size={64} />
              <p className="mt-3 text-sm text-gray-700">Bangladesh Legal Advisor</p>
              <h1 className="font-heading text-3xl font-bold text-gray-900">Bangladesh Legal Advisor</h1>
            </div>

            <div className="rounded-2xl border-2 border-white/60 bg-white/30 p-10 shadow-xl backdrop-blur-2xl text-center">
              <h2 className="font-heading text-xl font-bold text-gray-800">Invalid reset link</h2>
              <p className="mt-2 text-sm text-gray-700">
                This link is missing or malformed. Please request a new password reset link.
              </p>
              <Button
                type="button"
                className="mt-6 h-11 w-full rounded-xl text-base bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer"
                onClick={() => router.replace("/login")}
              >
                Back to login
              </Button>
            </div>
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
          <title>Password Reset Successful — Bangladesh Legal Advisor</title>
        </Head>
        <main className="relative min-h-dvh overflow-hidden bg-gradient-to-b from-[#e0f2fe] via-[#bae6fd] to-white flex items-center justify-center px-6 py-8">
          <div className="w-full max-w-xl animate-in fade-in zoom-in-95 duration-500">
            <div className="mb-8 flex flex-col items-center text-center">
              <BrandMark size={64} />
              <p className="mt-3 text-sm text-gray-700">Bangladesh Legal Advisor</p>
              <h1 className="font-heading text-3xl font-bold text-gray-900">Bangladesh Legal Advisor</h1>
            </div>

            <div className="rounded-2xl border-2 border-white/60 bg-white/30 p-10 shadow-xl backdrop-blur-2xl text-center">
              <span className="inline-flex size-16 items-center justify-center rounded-full bg-[#e0f2fe] text-[#0284c7] mb-4">
                <CheckCircle2 className="size-8" />
              </span>
              <h2 className="font-heading text-xl font-bold text-gray-800">Password Reset Successful</h2>
              <p className="mt-2 text-sm text-gray-700">
                {successMsg}
              </p>
              <Button
                type="button"
                className="mt-6 h-11 w-full rounded-xl text-base bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer"
                onClick={() => router.replace("/login")}
              >
                Go to login
              </Button>
            </div>
          </div>
        </main>
      </>
    )
  }

  const isSubmitting = pageState === "submitting"

  return (
    <>
      <Head>
        <title>Set New Password — Bangladesh Legal Advisor</title>
      </Head>
      <main className="relative min-h-dvh overflow-hidden bg-gradient-to-b from-[#e0f2fe] via-[#bae6fd] to-white flex items-center justify-center px-6 py-8">
        <div className="w-full max-w-xl animate-in fade-in zoom-in-95 duration-500">
          <div className="mb-8 flex flex-col items-center text-center">
            <BrandMark size={64} />
            <p className="mt-3 text-sm text-gray-700">Bangladesh Legal Advisor</p>
            <h1 className="font-heading text-3xl font-bold text-gray-900">Bangladesh Legal Advisor</h1>
          </div>

          <div className="rounded-2xl border-2 border-white/60 bg-white/30 p-10 sm:p-14 shadow-xl backdrop-blur-2xl hover:shadow-2xl transition-shadow duration-300">
            <h2 className="font-heading text-xl font-bold text-gray-800 text-center">
              Set new password
            </h2>
            <p className="mt-1 text-sm text-gray-700 text-center mb-6">
              Please enter your new password below.
            </p>

            {visibleError ? (
              <p className="mb-4 rounded-xl border border-destructive/25 bg-destructive/10 px-3.5 py-2.5 text-sm text-destructive">
                {visibleError}
              </p>
            ) : null}

            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="space-y-1.5">
                <label htmlFor="password" className="text-sm font-medium text-gray-800">
                  New Password
                </label>
                <div className="relative">
                  <Lock className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-gray-600" />
                  <input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    disabled={isSubmitting}
                    placeholder="Enter new password"
                    autoComplete="new-password"
                    className="h-11 w-full rounded-xl border-2 border-[#0284c7]/30 bg-white/20 pl-11 pr-10 text-sm text-gray-900 placeholder:text-gray-600 outline-none transition-all duration-300 backdrop-blur-sm disabled:cursor-not-allowed disabled:opacity-60 focus:border-[#0284c7] focus:bg-white/30 focus:ring-2 focus:ring-[#0284c7]/30 overflow-hidden text-ellipsis"
                  />
                  <button
                    type="button"
                    tabIndex={-1}
                    onClick={() => setShowPassword((p) => !p)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-600 transition hover:text-gray-900"
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                  </button>
                </div>
              </div>

              <div className="space-y-1.5">
                <label htmlFor="confirm-password" className="text-sm font-medium text-gray-800">
                  Confirm Password
                </label>
                <div className="relative">
                  <Lock className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-gray-600" />
                  <input
                    id="confirm-password"
                    type={showConfirm ? "text" : "password"}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    disabled={isSubmitting}
                    placeholder="Repeat new password"
                    autoComplete="new-password"
                    className="h-11 w-full rounded-xl border-2 border-[#0284c7]/30 bg-white/20 pl-11 pr-10 text-sm text-gray-900 placeholder:text-gray-600 outline-none transition-all duration-300 backdrop-blur-sm disabled:cursor-not-allowed disabled:opacity-60 focus:border-[#0284c7] focus:bg-white/30 focus:ring-2 focus:ring-[#0284c7]/30 overflow-hidden text-ellipsis"
                  />
                  <button
                    type="button"
                    tabIndex={-1}
                    onClick={() => setShowConfirm((p) => !p)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-600 transition hover:text-gray-900"
                    aria-label={showConfirm ? "Hide password" : "Show password"}
                  >
                    {showConfirm ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                  </button>
                </div>
                {confirmPassword.length > 0 ? (
                  <p className={`text-xs ${password === confirmPassword ? "text-[#0284c7]" : "text-destructive"}`}>
                    {password === confirmPassword ? "✓ Passwords match" : "✗ Passwords do not match"}
                  </p>
                ) : null}
              </div>

              <Button
                type="submit"
                size="lg"
                disabled={isSubmitting}
                className="h-11 w-full rounded-xl text-base bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer mt-2"
              >
                <span className="inline-flex size-4 items-center justify-center">
                  {isSubmitting ? <Loader2 className="size-4 animate-spin" /> : null}
                </span>
                Reset password
              </Button>
            </form>
          </div>

          <div className="absolute bottom-0 left-0 right-0 pb-4 text-center">
            <p className="text-xs font-bold text-gray-500">
              © {new Date().getFullYear()} Bangladesh Legal Advisor. All rights reserved.
            </p>
          </div>
        </div>
      </main>
    </>
  )
}
