import { apiRequest } from "@/lib/http"
import type { HealthResponse } from "@/lib/types"

export const healthApi = {
  check() {
    return apiRequest<HealthResponse>({
      method: "GET",
      url: "/health",
    })
  },
}
