import { apiRequest } from "@/lib/http"
import type { ChatRequest, ChatResponse } from "@/lib/types"

export const chatApi = {
  ask(payload: ChatRequest) {
    return apiRequest<ChatResponse>({
      method: "POST",
      url: "/chat",
      data: payload,
    })
  },
}
