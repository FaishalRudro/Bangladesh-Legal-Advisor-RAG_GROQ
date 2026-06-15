import { useCallback, useEffect, useState } from "react"
import { healthApi } from "@/lib/api/health"
import type { ServerState } from "@/lib/app-types"

export function useHealth(enabled: boolean) {
  const [serverState, setServerState] = useState<ServerState>("checking")

  const loadHealth = useCallback(async () => {
    setServerState("checking")

    try {
      await healthApi.check()
      setServerState("online")
    } catch {
      setServerState("offline")
    }
  }, [])

  useEffect(() => {
    if (!enabled) {
      setServerState("checking")
      return
    }

    loadHealth()
  }, [enabled, loadHealth])

  return {
    serverState,
    refreshHealth: loadHealth,
  }
}
