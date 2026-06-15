import { apiRequest } from "@/lib/http"
import type { LawListResponse, LawIngestResponse } from "@/lib/types"

export const lawsApi = {
  list(page: number = 1, size: number = 10) {
    return apiRequest<LawListResponse>({
      method: "GET",
      url: "/laws",
      params: { page, size },
    })
  },

  ingest(limit: number = 0) {
    return apiRequest<LawIngestResponse>({
      method: "POST",
      url: `/laws/ingest?limit=${limit}`,
    })
  },

  delete(lawId: string) {
    return apiRequest<void>({
      method: "DELETE",
      url: `/laws/${encodeURIComponent(lawId)}`,
    })
  },
}
