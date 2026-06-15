import { z } from "zod"

// Minimal checks only — just prevents submitting empty fields.
// All real validation (password strength, email format, OTP format, etc.)
// is handled by the backend.

export const loginSchema = z.object({
  email: z.string().min(1, "Email is required."),
  password: z.string().min(1, "Password is required."),
})

export const signupSchema = z.object({
  name: z.string().min(1, "Name is required."),
  email: z.string().min(1, "Email is required."),
  password: z.string().min(1, "Password is required."),
})

export const otpSchema = z.object({
  email: z.string().min(1, "Email is required."),
  otp: z.string().min(1, "OTP is required."),
})

export const resendOtpSchema = z.object({
  email: z.string().min(1, "Email is required."),
})

export const forgotPasswordSchema = z.object({
  email: z.string().min(1, "Email is required."),
})

export const resetPasswordSchema = z.object({
  token: z.string().min(1, "Token is required."),
  new_password: z.string().min(1, "Password is required."),
  confirmPassword: z.string().min(1, "Please confirm your password."),
}).refine((data) => data.new_password === data.confirmPassword, {
  message: "Passwords do not match.",
  path: ["confirmPassword"],
})

export const acceptInviteSchema = z.object({
  email: z.string().min(1, "Email is required."),
  password: z.string().min(1, "Password is required."),
  confirmPassword: z.string().min(1, "Please confirm your password."),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords do not match.",
  path: ["confirmPassword"],
})

export type LoginFormValues = z.infer<typeof loginSchema>
export type SignupFormValues = z.infer<typeof signupSchema>
export type OtpFormValues = z.infer<typeof otpSchema>
export type AcceptInviteFormValues = z.infer<typeof acceptInviteSchema>
export type ForgotPasswordFormValues = z.infer<typeof forgotPasswordSchema>
export type ResetPasswordFormValues = z.infer<typeof resetPasswordSchema>

export function getValidationMessage(error: unknown) {
  if (error instanceof z.ZodError) {
    return error.issues[0]?.message ?? "Please fill in all required fields."
  }

  return "Please fill in all required fields."
}
