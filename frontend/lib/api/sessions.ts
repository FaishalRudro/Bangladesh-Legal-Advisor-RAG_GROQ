import { apiRequest } from "@/lib/http";
import type { SessionListResponse, SessionDetailResponse } from "@/lib/types";

export const sessionsApi = {
  getSessions(page = 1, size = 50) {
    return apiRequest<SessionListResponse>({
      method: "GET",
      url: "/sessions",
      params: { page, size },
    });
  },

  getSession(sessionId: number, page = 1, size = 50) {
    return apiRequest<SessionDetailResponse>({
      method: "GET",
      url: `/sessions/${sessionId}`,
      params: { page, size },
    });
  },

  deleteSession(sessionId: number) {
    return apiRequest<string>({
      method: "DELETE",
      url: `/sessions/${sessionId}`,
    });
  },

  pinSession(sessionId: number, isPinned: boolean) {
    return apiRequest<string>({
      method: "PATCH",
      url: `/sessions/${sessionId}/pin`,
      data: { is_pinned: isPinned },
    });
  },

  submitFeedback(messageId: number, isGood: boolean, feedbackText?: string) {
    return apiRequest<string>({
      method: "POST",
      url: `/messages/${messageId}/feedback`,
      data: { is_good: isGood, feedback_text: feedbackText },
    });
  },
};
