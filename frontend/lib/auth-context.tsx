import { createContext, useContext, type ReactNode } from "react"
import { useAuth } from "@/hooks/use-auth"
import type { AuthSession } from "@/lib/types"
import type { LoginFormValues, OtpFormValues, SignupFormValues } from "@/lib/validation/auth"

type AuthContextValue = {
  session: AuthSession | null
  isRestoring: boolean
  isSubmitting: boolean
  isVerifying: boolean
  isResending: boolean
  error: string
  clearError: () => void
  login: (payload: LoginFormValues) => Promise<AuthSession>
  signup: (payload: SignupFormValues) => Promise<string>
  verifyOtp: (payload: OtpFormValues) => Promise<unknown>
  resendOtp: (email: string) => Promise<string>
  forgotPassword: (email: string) => Promise<string>
  logout: () => void
  fetchSession: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const auth = useAuth()
  return <AuthContext.Provider value={auth}>{children}</AuthContext.Provider>
}

export function useAuthContext(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuthContext must be used inside <AuthProvider>")
  return ctx
}
