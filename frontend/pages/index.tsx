import { useEffect } from "react"
import { useRouter } from "next/router"
import { useAuthContext } from "@/lib/auth-context"
import { LandingPage } from "@/components/landing-page"
import type { UserRole } from "@/lib/types"

const roleHome: Record<UserRole, string> = {
  super_admin: "/admin",
  Lawyer: "/Lawyer",
  user: "/user",
}

/**
 * Root route — shows landing page if not logged in, redirects to role dashboard if logged in.
 */
export default function RootPage() {
  const { session, isRestoring } = useAuthContext()
  const router = useRouter()

  useEffect(() => {
    if (isRestoring) return
    if (session) {
      router.replace(roleHome[session.user.role])
    }
  }, [isRestoring, session, router])

  if (isRestoring) {
    return (
      <main className="grid min-h-screen place-items-center bg-gradient-to-b from-[#e0f2fe] via-[#bae6fd] to-white text-sm text-gray-700">
        Loading...
      </main>
    )
  }

  if (session) {
    return null // Will redirect via useEffect
  }

  return <LandingPage />
}
