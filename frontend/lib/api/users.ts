import { apiRequest } from "@/lib/http"
import type { UserResponse, ForgotPassword, ResetPassword } from "@/lib/types"

export const usersApi = {
  getMe() {
    return apiRequest<UserResponse>({
      method: "GET",
      url: "/users/me",
    })
  },

  forgotPassword(payload: ForgotPassword) {
    return apiRequest<string>({
      method: "POST",
      url: "/users/forgot-password",
      data: payload,
    })
  },

  resetPassword(payload: ResetPassword) {
    return apiRequest<string>({
      method: "POST",
      url: "/users/reset-password",
      data: payload,
    })
  },

  updateProfile(name: string) {
    return apiRequest<UserResponse>({
      method: "PATCH",
      url: "/users/me",
      data: { name },
    })
  },

  requestPasswordChange(oldPassword: string) {
    return apiRequest<{ detail: string }>({
      method: "POST",
      url: "/users/change-password/request",
      data: { old_password: oldPassword },
    })
  },

  verifyPasswordChange(otp: string, newPassword: string) {
    return apiRequest<{ detail: string }>({
      method: "POST",
      url: "/users/change-password/verify",
      data: { otp, new_password: newPassword },
    })
  },
}
