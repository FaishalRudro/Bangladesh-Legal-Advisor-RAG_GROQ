import { useState, useEffect, type FormEvent } from "react"
import { BrandMark } from "@/components/brand-mark"
import { Button } from "@/components/ui/button"
import type { AuthSession } from "@/lib/types"
import type { LoginFormValues, OtpFormValues, SignupFormValues, ForgotPasswordFormValues } from "@/lib/validation/auth"
import { getValidationMessage, loginSchema, otpSchema, resendOtpSchema, signupSchema, forgotPasswordSchema } from "@/lib/validation/auth"
import { useLanguage } from "@/lib/language-context"
import {
  ArrowLeft,
  BookOpen,
  Eye,
  EyeOff,
  Loader2,
  Lock,
  Mail,
  MessageCircleQuestion,
  ShieldCheck,
  Sparkles,
  User,
} from "lucide-react"

const translations = {
  bn: {
    foundation: "Bangladesh Legal Advisor",
    appName: "Legal Advisor Platform",
    aiAssistant: "এআই চালিত আইনি সহায়ক",
    trustKnowledge: "বিশ্বাসযোগ্য আইনি জ্ঞান যাচাইকৃত প্রবেশাধিকারে",
    signInDesc: "প্রশ্ন করতে, রেফারেন্স পর্যালোচনা করতে এবং আপনার আগের কথোপকথন চালিয়ে যেতে সাইন ইন করুন।",
    highlight1Title: "স্পষ্টভাবে প্রশ্ন করুন",
    highlight1Desc: "বাংলা বা ইংরেজিতে প্রশ্ন জমা দিন এবং উল্লেখিত সূত্র অনুসরণ করুন।",
    highlight2Title: "বিশ্বস্ত রেফারেন্স",
    highlight2Desc: "আইনি লাইব্রেরির সাথে যুক্ত তথ্যসূত্র থেকে উত্তর প্রদান করা হয়।",
    highlight3Title: "যাচাইকৃত প্রবেশাধিকার",
    highlight3Desc: "নতুন অ্যাকাউন্ট ইমেইল ওটিপি দিয়ে সাইন ইন করার আগে যাচাই করা হয়।",
    signIn: "সাইন ইন",
    signUp: "সাইন আপ",
    greeting: "স্বাগতম",
    createAccount: "আপনার অ্যাকাউন্ট তৈরি করুন",
    verifyOtp: "ওটিপি যাচাই করুন",
    useEmailPassword: "আপনার নিবন্ধিত ইমেইল এবং পাসওয়ার্ড ব্যবহার করুন।",
    enterDetails: "আপনার বিবরণ লিখুন। আমরা আপনার ইমেইলে একটি ৬-সংখ্যার ওটিপি পাঠাব।",
    otpSent: "আমরা",
    otpSentEnd: "একটি ওটিপি পাঠিয়েছি।",
    email: "ইমেইল",
    password: "পাসওয়ার্ড",
    name: "নাম",
    otp: "ওটিপি",
    emailPlaceholder: "আপনার ইমেইল",
    passwordPlaceholder: "আপনার পাসওয়ার্ড লিখুন",
    createPasswordPlaceholder: "একটি পাসওয়ার্ড তৈরি করুন",
    namePlaceholder: "আপনার পুরো নাম",
    otpPlaceholder: "৬-সংখ্যার কোড",
    sendOtp: "ওটিপি পাঠান",
    verifyAccount: "অ্যাকাউন্ট যাচাই করুন",
    resendOtp: "পুনরায় ওটিপি পাঠান",
    backToSignup: "সাইন আপ এ ফিরে যান",
    forgotPassword: "পাসওয়ার্ড ভুলে গেছেন?",
    sendResetLink: "রিসেট লিংক পাঠান",
    forgotPasswordDesc: "আপনার ইমেইল দিন, আমরা পাসওয়ার্ড রিসেটের লিংক পাঠাব।",
    backToSignin: "সাইন ইন এ ফিরে যান",
    copyright: "© {year} Bangladesh Legal Advisor. All rights reserved.",
  },
  en: {
    foundation: "Bangladesh Legal Advisor",
    appName: "Legal Advisor Platform",
    aiAssistant: "AI-Powered Legal Assistant",
    trustKnowledge: "Trusted Legal Knowledge with Verified Access",
    signInDesc: "Sign in to ask questions, review references, and continue your previous conversations.",
    highlight1Title: "Ask Questions Clearly",
    highlight1Desc: "Submit questions in Bengali or English and follow the cited sources.",
    highlight2Title: "Trusted References",
    highlight2Desc: "Answers provided from sources linked to the Legal knowledge library.",
    highlight3Title: "Verified Access",
    highlight3Desc: "New accounts are verified with email OTP before signing in.",
    signIn: "Sign In",
    signUp: "Sign Up",
    greeting: "Dear User",
    createAccount: "Create Your Account",
    verifyOtp: "Verify OTP",
    useEmailPassword: "Use your registered email and password.",
    enterDetails: "Enter your details. We will send a 6-digit OTP to your email.",
    otpSent: "We sent an OTP to",
    otpSentEnd: "",
    email: "Email",
    password: "Password",
    name: "Name",
    otp: "OTP",
    emailPlaceholder: "your email",
    passwordPlaceholder: "Enter your password",
    createPasswordPlaceholder: "Create a password",
    namePlaceholder: "Your full name",
    otpPlaceholder: "6-digit code",
    sendOtp: "Send OTP",
    verifyAccount: "Verify Account",
    resendOtp: "Resend OTP",
    backToSignup: "Back to Sign Up",
    forgotPassword: "Forgot Password?",
    sendResetLink: "Send Reset Link",
    forgotPasswordDesc: "Enter your email and we'll send a password reset link.",
    backToSignin: "Back to Sign In",
    copyright: "© {year} Bangladesh Legal Advisor. All rights reserved.",
  },
}

type AuthMode = "signin" | "signup" | "otp" | "forgot-password"

type LoginScreenProps = {
  apiError: string
  isSubmitting: boolean
  isVerifying: boolean
  isResending: boolean
  clearError: () => void
  onLogin: (payload: LoginFormValues) => Promise<AuthSession>
  onSignup: (payload: SignupFormValues) => Promise<string>
  onVerifyOtp: (payload: OtpFormValues) => Promise<unknown>
  onResendOtp: (email: string) => Promise<string>
  onForgotPassword: (email: string) => Promise<string>
}

export function LoginScreen({
  apiError,
  isSubmitting,
  isVerifying,
  isResending,
  clearError,
  onLogin,
  onSignup,
  onVerifyOtp,
  onResendOtp,
  onForgotPassword,
}: LoginScreenProps) {
  const { language } = useLanguage()
  const t = translations[language]

  const highlights = [
    {
      icon: MessageCircleQuestion,
      title: t.highlight1Title,
      desc: t.highlight1Desc,
    },
    {
      icon: BookOpen,
      title: t.highlight2Title,
      desc: t.highlight2Desc,
    },
    {
      icon: ShieldCheck,
      title: t.highlight3Title,
      desc: t.highlight3Desc,
    },
  ]

  const [mode, setMode] = useState<AuthMode>("signin")
  const [signinForm, setSigninForm] = useState<LoginFormValues>({ email: "", password: "" })
  const [signupForm, setSignupForm] = useState<SignupFormValues>({ name: "", email: "", password: "" })
  const [otpForm, setOtpForm] = useState<OtpFormValues>({ email: "", otp: "" })
  const [forgotPasswordForm, setForgotPasswordForm] = useState<ForgotPasswordFormValues>({ email: "" })
  const [formError, setFormError] = useState("")
  const [message, setMessage] = useState("")
  const [countdown, setCountdown] = useState(0)

  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000)
      return () => clearTimeout(timer)
    }
  }, [countdown])

  const visibleError = formError || apiError
  const isBusy = isSubmitting || isVerifying || isResending

  function resetFeedback() {
    setFormError("")
    setMessage("")
    clearError()
  }

  function switchMode(nextMode: AuthMode) {
    resetFeedback()
    setMode(nextMode)
  }

  async function handleSignin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    resetFeedback()

    const result = loginSchema.safeParse(signinForm)
    if (!result.success) {
      setFormError(getValidationMessage(result.error))
      return
    }

    try {
      await onLogin(result.data)
    } catch {
      // The auth hook exposes backend errors through apiError.
    }
  }

  async function handleSignup(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    resetFeedback()

    const result = signupSchema.safeParse(signupForm)
    if (!result.success) {
      setFormError(getValidationMessage(result.error))
      return
    }

    try {
      const backendMsg = await onSignup(result.data)
      setOtpForm({ email: result.data.email, otp: "" })
      setMode("otp")
      setMessage(typeof backendMsg === "string" ? backendMsg : "")
      setCountdown(90)
    } catch {
      // The auth hook exposes backend errors through apiError.
    }
  }

  async function handleVerifyOtp(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    resetFeedback()

    const result = otpSchema.safeParse(otpForm)
    if (!result.success) {
      setFormError(getValidationMessage(result.error))
      return
    }

    try {
      const verifyRes = await onVerifyOtp(result.data)
      setSigninForm((current) => ({ ...current, email: result.data.email, password: "" }))
      setOtpForm({ email: result.data.email, otp: "" })
      setMode("signin")
      setMessage((verifyRes as any)?.message || "")
    } catch {
      // The auth hook exposes backend errors through apiError.
    }
  }

  async function handleResendOtp() {
    resetFeedback()

    const result = resendOtpSchema.safeParse({ email: otpForm.email })
    if (!result.success) {
      setFormError(getValidationMessage(result.error))
      return
    }

    try {
      const backendMsg = await onResendOtp(result.data.email)
      setMessage(typeof backendMsg === "string" ? backendMsg : "")
      setCountdown(90)
    } catch {
      // The auth hook exposes backend errors through apiError.
    }
  }

  async function handleForgotPassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    resetFeedback()

    const result = forgotPasswordSchema.safeParse(forgotPasswordForm)
    if (!result.success) {
      setFormError(getValidationMessage(result.error))
      return
    }

    try {
      const backendMsg = await onForgotPassword(result.data.email)
      const msgText = typeof backendMsg === "string" ? backendMsg : (backendMsg as any)?.detail || "Password reset link sent to your email."
      setMessage(msgText)
      setForgotPasswordForm({ email: "" })
    } catch {
      // The auth hook exposes backend errors through apiError.
    }
  }

  return (
    <main className="relative min-h-dvh overflow-hidden bg-gradient-to-b from-[#e0f2fe] via-[#bae6fd] to-white flex items-center justify-center px-6 py-8">

      {/* Centered login form */}
      <div className="w-full max-w-xl animate-in fade-in zoom-in-95 duration-500">
        <div className="mb-8 flex flex-col items-center text-center">
          <BrandMark size={64} />
          <p className="mt-3 text-sm text-gray-700">{t.foundation}</p>
          <h1 className="font-heading text-3xl font-bold text-gray-900">{t.appName}</h1>
        </div>

        <div className="rounded-2xl border-2 border-white/60 bg-white/30 p-20 shadow-xl backdrop-blur-2xl hover:shadow-2xl transition-shadow duration-300">
          {mode === "otp" || mode === "forgot-password" ? (
            <button
              type="button"
              onClick={() => switchMode(mode === "otp" ? "signup" : "signin")}
              className="mb-4 inline-flex items-center gap-1.5 text-sm font-medium text-gray-700 transition hover:text-gray-900 cursor-pointer"
            >
              <ArrowLeft className="size-4" />
              {mode === "otp" ? t.backToSignup : t.backToSignin}
            </button>
          ) : (
            <div className="mb-5 grid grid-cols-2 rounded-xl border border-white/30 bg-white/15 p-1 backdrop-blur-sm">
              <button
                type="button"
                onClick={() => switchMode("signin")}
                className={`rounded-lg px-3 py-2 text-sm font-semibold cursor-pointer ${mode === "signin" ? "bg-white/30 text-gray-900 shadow-sm backdrop-blur-sm border-2 border-[#0284c7]" : "text-gray-700 hover:text-gray-900"
                  }`}
              >
                {t.signIn}
              </button>
              <button
                type="button"
                onClick={() => switchMode("signup")}
                className={`rounded-lg px-3 py-2 text-sm font-semibold cursor-pointer ${mode === "signup" ? "bg-white/30 text-gray-900 shadow-sm backdrop-blur-sm border-2 border-[#0284c7]" : "text-gray-700 hover:text-gray-900"
                  }`}
              >
                {t.signUp}
              </button>
            </div>
          )}

          <h2 className="font-heading text-xl font-bold text-gray-800 text-center">
            {mode === "signin" ? t.greeting : mode === "signup" ? t.createAccount : mode === "forgot-password" ? t.forgotPassword : t.verifyOtp}
          </h2>
          <p className="mt-1 text-sm text-gray-700 text-center">
            {mode === "signin"
              ? t.useEmailPassword
              : mode === "signup"
                ? t.enterDetails
                : mode === "forgot-password"
                  ? t.forgotPasswordDesc
                  : `${t.otpSent} ${otpForm.email || language === "bn" ? "আপনার ইমেইলে" : "your email"}${t.otpSentEnd ? ` ${t.otpSentEnd}` : "."}`}
          </p>

          {visibleError ? (
            <p className="mt-4 rounded-xl border border-destructive/25 bg-destructive/10 px-3.5 py-2.5 text-sm text-destructive">
              {visibleError}
            </p>
          ) : null}

          {message ? (
            <p className="mt-4 rounded-xl border border-primary/20 bg-accent px-3.5 py-2.5 text-sm text-foreground">
              {message}
            </p>
          ) : null}

          <div className="relative overflow-hidden px-4">
            {mode === "signin" ? (
              <form key="signin" className="mt-6 space-y-4 animate-in fade-in duration-700 [animation-timing-function:cubic-bezier(0.4,0,0.2,1)]" style={{ animation: 'ripple 0.7s ease-out' }} onSubmit={handleSignin}>
                <TextField
                  id="signin-email"
                  label={t.email}
                  type="email"
                  icon={Mail}
                  placeholder={t.emailPlaceholder}
                  value={signinForm.email}
                  onChange={(value) => setSigninForm((current) => ({ ...current, email: value }))}
                  disabled={isBusy}
                  autoComplete="email"
                />
                <TextField
                  id="signin-password"
                  label={t.password}
                  type="password"
                  icon={Lock}
                  placeholder={t.passwordPlaceholder}
                  value={signinForm.password}
                  onChange={(value) => setSigninForm((current) => ({ ...current, password: value }))}
                  disabled={isBusy}
                  autoComplete="current-password"
                />

                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => switchMode("forgot-password")}
                    className="text-sm font-medium text-[#0284c7] hover:underline"
                  >
                    {t.forgotPassword}
                  </button>
                </div>

                <Button type="submit" size="lg" disabled={isSubmitting} className="h-11 w-full rounded-xl text-base bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer">
                  <span className="inline-flex size-4 items-center justify-center">
                    {isSubmitting ? <Loader2 className="size-4 animate-spin" /> : null}
                  </span>
                  {t.signIn}
                </Button>
              </form>
            ) : null}

            {mode === "signup" ? (
              <form key="signup" className="mt-6 space-y-4 animate-in fade-in duration-700" style={{ animation: 'ripple 0.7s ease-out' }} onSubmit={handleSignup}>
                <TextField
                  id="signup-name"
                  label={t.name}
                  type="text"
                  icon={User}
                  placeholder={t.namePlaceholder}
                  value={signupForm.name}
                  onChange={(value) => setSignupForm((current) => ({ ...current, name: value }))}
                  disabled={isBusy}
                  autoComplete="name"
                />
                <TextField
                  id="signup-email"
                  label={t.email}
                  type="email"
                  icon={Mail}
                  placeholder={t.emailPlaceholder}
                  value={signupForm.email}
                  onChange={(value) => setSignupForm((current) => ({ ...current, email: value }))}
                  disabled={isBusy}
                  autoComplete="email"
                />
                <TextField
                  id="signup-password"
                  label={t.password}
                  type="password"
                  icon={Lock}
                  placeholder={t.createPasswordPlaceholder}
                  value={signupForm.password}
                  onChange={(value) => setSignupForm((current) => ({ ...current, password: value }))}
                  disabled={isBusy}
                  autoComplete="new-password"
                />

                <Button type="submit" size="lg" disabled={isSubmitting} className="h-11 w-full rounded-xl text-base bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer">
                  <span className="inline-flex size-4 items-center justify-center">
                    {isSubmitting ? <Loader2 className="size-4 animate-spin" /> : null}
                  </span>
                  {t.sendOtp}
                </Button>
              </form>
            ) : null}

            {mode === "otp" ? (
              <form key="otp" className="mt-6 space-y-4 animate-in fade-in duration-700" style={{ animation: 'ripple 0.7s ease-out' }} onSubmit={handleVerifyOtp}>
                <TextField
                  id="otp-email"
                  label={t.email}
                  type="email"
                  icon={Mail}
                  placeholder={t.emailPlaceholder}
                  value={otpForm.email}
                  onChange={(value) => setOtpForm((current) => ({ ...current, email: value }))}
                  disabled={isBusy}
                  autoComplete="email"
                />
                <TextField
                  id="otp-code"
                  label={t.otp}
                  type="text"
                  icon={ShieldCheck}
                  placeholder={t.otpPlaceholder}
                  value={otpForm.otp}
                  onChange={(value) => setOtpForm((current) => ({ ...current, otp: value.replace(/\D/g, "").slice(0, 6) }))}
                  disabled={isBusy}
                  autoComplete="one-time-code"
                  inputMode="numeric"
                  maxLength={6}
                />

                <Button type="submit" size="lg" disabled={isVerifying} className="h-11 w-full rounded-xl text-base bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer">
                  <span className="inline-flex size-4 items-center justify-center">
                    {isVerifying ? <Loader2 className="size-4 animate-spin" /> : null}
                  </span>
                  {t.verifyAccount}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  disabled={isResending || countdown > 0}
                  onClick={handleResendOtp}
                  className="h-10 w-full rounded-xl cursor-pointer"
                >
                  <span className="inline-flex size-4 items-center justify-center">
                    {isResending ? <Loader2 className="size-4 animate-spin" /> : null}
                  </span>
                  {countdown > 0 ? `${t.resendOtp} (${countdown}s)` : t.resendOtp}
                </Button>
              </form>
            ) : null}

            {mode === "forgot-password" ? (
              <form key="forgot-password" className="mt-6 space-y-4 animate-in fade-in duration-700" style={{ animation: 'ripple 0.7s ease-out' }} onSubmit={handleForgotPassword}>
                <TextField
                  id="forgot-email"
                  label={t.email}
                  type="email"
                  icon={Mail}
                  placeholder={t.emailPlaceholder}
                  value={forgotPasswordForm.email}
                  onChange={(value) => setForgotPasswordForm({ email: value })}
                  disabled={isBusy}
                  autoComplete="email"
                />

                <Button type="submit" size="lg" disabled={isSubmitting} className="h-11 w-full rounded-xl text-base bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer">
                  <span className="inline-flex size-4 items-center justify-center">
                    {isSubmitting ? <Loader2 className="size-4 animate-spin" /> : null}
                  </span>
                  {t.sendResetLink}
                </Button>
              </form>
            ) : null}
          </div>
        </div>
      </div>

      {/* Copyright footer - bottom center */}
      <div className="absolute bottom-0 left-0 right-0 pb-4 text-center">
        <p className="text-xs font-bold text-gray-500">
          {t.copyright.replace("{year}", new Date().getFullYear().toString())}
        </p>
      </div>
    </main>
  )
}

function TextField({
  id,
  label,
  type,
  icon: Icon,
  placeholder,
  value,
  onChange,
  disabled,
  autoComplete,
  inputMode,
  maxLength,
}: {
  id: string
  label: string
  type: string
  icon: typeof Mail
  placeholder?: string
  value: string
  onChange: (value: string) => void
  disabled: boolean
  autoComplete: string
  inputMode?: "email" | "numeric" | "search" | "tel" | "text" | "url"
  maxLength?: number
}) {
  const [showPassword, setShowPassword] = useState(false)
  const isPassword = type === "password"
  const resolvedType = isPassword ? (showPassword ? "text" : "password") : type

  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="text-sm font-medium text-gray-800">
        {label}
      </label>
      <div className="relative">
        <Icon className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-gray-600" />
        <input
          id={id}
          type={resolvedType}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
          autoComplete={autoComplete}
          inputMode={inputMode}
          maxLength={maxLength}
          placeholder={placeholder}
          className="h-11 w-full rounded-xl border-2 border-[#0284c7]/30 bg-white/20 pl-11 pr-10 text-sm text-gray-900 placeholder:text-gray-600 outline-none transition-all duration-300 backdrop-blur-sm disabled:cursor-not-allowed disabled:opacity-60 focus:border-[#0284c7] focus:bg-white/30 focus:ring-2 focus:ring-[#0284c7]/30 overflow-hidden text-ellipsis"
        />
        {isPassword ? (
          <button
            type="button"
            onClick={() => setShowPassword((prev) => !prev)}
            className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-600 transition hover:text-gray-900"
            aria-label={showPassword ? "Hide password" : "Show password"}
            tabIndex={-1}
          >
            {showPassword ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        ) : null}
      </div>
    </div>
  )
}
