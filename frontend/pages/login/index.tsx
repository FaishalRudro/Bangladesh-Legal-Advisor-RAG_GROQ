import { useEffect } from "react"
import { useRouter } from "next/router"
import Head from "next/head"
import { LoginScreen } from "@/components/login-screen"
import { useAuthContext } from "@/lib/auth-context"
import type { UserRole } from "@/lib/types"

const roleHome: Record<UserRole, string> = {
  super_admin: "/admin",
  Lawyer: "/Lawyer",
  user: "/user",
}

export default function LoginPage() {
  const { session, isRestoring, isSubmitting, isVerifying, isResending, error, clearError, login, signup, verifyOtp, resendOtp, forgotPassword } =
    useAuthContext()
  const router = useRouter()

  // Already logged in — send to the right dashboard
  useEffect(() => {
    if (!isRestoring && session) {
      router.replace(roleHome[session.user.role])
    }
  }, [isRestoring, session, router])

  if (isRestoring || session) {
    return (
      <main className="grid min-h-screen place-items-center bg-gradient-to-b from-[#e0f2fe] via-[#bae6fd] to-white text-sm text-gray-700">
        Loading...
      </main>
    )
  }

  return (
    <>
      <Head>
        <title>Login — Bangladesh Legal Advisor</title>
      </Head>
      <LoginScreen
        apiError={error}
        isSubmitting={isSubmitting}
        isVerifying={isVerifying}
        isResending={isResending}
        clearError={clearError}
        onLogin={async (payload) => {
          const session = await login(payload)
          router.replace(roleHome[session.user.role])
          return session
        }}
        onSignup={signup}
        onVerifyOtp={verifyOtp}
        onResendOtp={resendOtp}
        onForgotPassword={forgotPassword}
      />
    </>
  )
}
