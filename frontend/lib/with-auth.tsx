import { useRouter } from "next/router"
import { useEffect, type ComponentType } from "react"
import { useAuthContext } from "@/lib/auth-context"
import type { UserRole } from "@/lib/types"
import NotFoundPage from "@/pages/404"

/**
 * HOC that protects a page by role.
 *
 * Usage:
 *   export default withAuth(MyPage, ["super_admin"])
 *   export default withAuth(MyPage, ["super_admin", "Lawyer", "user"])
 */
export function withAuth<P extends object>(
  WrappedComponent: ComponentType<P>,
  allowedRoles: UserRole[],
) {
  function ProtectedPage(props: P) {
    const { session, isRestoring } = useAuthContext()
    const router = useRouter()

    useEffect(() => {
      if (isRestoring) return

      if (!session) {
        router.replace("/login")
        return
      }
    }, [isRestoring, session, router])

    // Show nothing while restoring session or redirecting
    if (isRestoring || !session) {
      return (
        <main className="grid min-h-screen place-items-center bg-gradient-to-b from-[#e0f2fe] via-[#bae6fd] to-white text-sm text-gray-700">
          Loading...
        </main>
      )
    }

    if (!allowedRoles.includes(session.user.role)) {
      return <NotFoundPage />
    }

    return <WrappedComponent {...props} />
  }

  ProtectedPage.displayName = `withAuth(${WrappedComponent.displayName ?? WrappedComponent.name})`
  return ProtectedPage
}
