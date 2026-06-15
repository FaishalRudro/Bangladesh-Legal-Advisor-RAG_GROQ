import { useState, type FormEvent } from "react"
import { authApi } from "@/lib/api/auth"
import { getApiErrorMessage } from "@/lib/http"
import type { UserResponse } from "@/lib/types"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { CheckCircle2, Loader2, Mail, Send, UserPlus, UserRound } from "lucide-react"

type FormValues = { name: string; email: string }
type FormState = "idle" | "submitting" | "success"

export function CreateLawyerDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const [form, setForm] = useState<FormValues>({ name: "", email: "" })
  const [formState, setFormState] = useState<FormState>("idle")
  const [createdUser, setCreatedUser] = useState<UserResponse | null>(null)
  const [error, setError] = useState("")

  function resetForm() {
    setForm({ name: "", email: "" })
    setFormState("idle")
    setCreatedUser(null)
    setError("")
  }

  function handleOpenChange(nextOpen: boolean) {
    if (nextOpen && !open) {
      resetForm()
    }

    onOpenChange(nextOpen)
  }

  function handleChange(field: keyof FormValues, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }))
    setError("")
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError("")

    const name = form.name.trim()
    const email = form.email.trim()

    if (name.length < 2) {
      setError("নাম অন্তত ২ অক্ষরের হতে হবে।")
      return
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("একটি সঠিক ইমেইল ঠিকানা দিন।")
      return
    }

    setFormState("submitting")

    try {
      const user = await authApi.createLawyer({ name, email })
      setCreatedUser(user)
      setFormState("success")
      setForm({ name: "", email: "" })
    } catch (err) {
      setError(getApiErrorMessage(err))
      setFormState("idle")
    }
  }

  function handleCreateAnother() {
    setCreatedUser(null)
    setFormState("idle")
    setError("")
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-hidden rounded-lg border border-border bg-popover p-0 shadow-2xl sm:max-w-[520px] data-open:duration-200 data-open:slide-in-from-bottom-3">
        <div className="relative border-b border-border bg-secondary/70 px-5 py-5 sm:px-6">
          <div className="absolute inset-x-0 top-0 h-1 bg-primary" />
          <DialogHeader className="pr-8">
            <div className="flex items-center gap-3">
              <span className="inline-flex size-11 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
                <UserPlus className="size-5" />
              </span>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-primary">
                  অ্যাডমিন আমন্ত্রণ
                </p>
                <DialogTitle className="mt-1 font-heading text-xl font-bold leading-tight">
                  লইয়ার আমন্ত্রণ করুন
                </DialogTitle>
              </div>
            </div>
            <DialogDescription className="max-w-[28rem] pt-1 leading-relaxed">
              নতুন লইয়ারের জন্য অ্যাকাউন্ট তৈরি এবং পাসওয়ার্ড সেটআপের জন্য ইমেইল স্বয়ংক্রিয়ভাবে পাঠান
            </DialogDescription>
          </DialogHeader>
        </div>

        {formState === "success" && createdUser ? (
          <div className="animate-fade-rise p-5 sm:p-6">
            <div className="text-center">
              <span className="inline-flex size-16 items-center justify-center rounded-full bg-accent text-primary ring-8 ring-accent/45">
                <CheckCircle2 className="size-8" />
              </span>
              <h2 className="mt-5 font-heading text-lg font-bold">আমন্ত্রণ পাঠানো হয়েছে</h2>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                এই লইয়ারের ইমেইলে পাসওয়ার্ড সেটআপ লিংকসহ ওয়েলকাম ইমেইল পাঠানো হয়েছে।
              </p>
              <p className="mt-3 break-all rounded-lg border border-border bg-muted px-4 py-2.5 font-mono text-sm font-medium text-foreground">
                {createdUser.email}
              </p>
            </div>

            <div className="mt-5 grid gap-2 rounded-lg border border-border bg-secondary/45 p-4 text-sm">
              <p>
                <span className="font-medium">নাম:</span> {createdUser.name}
              </p>
              <p>
                <span className="font-medium">ভূমিকা:</span> লইয়ার
              </p>
              <p>
                <span className="font-medium">স্ট্যাটাস:</span>{" "}
                <span className="text-muted-foreground">
                  {createdUser.is_verified ? "যাচাইকৃত" : "পাসওয়ার্ড সেটআপের অপেক্ষায়"}
                </span>
              </p>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={handleCreateAnother}
                className="h-10 w-full rounded-lg"
              >
                <UserPlus className="size-4" />
                আরেকজন
              </Button>
              <Button
                type="button"
                onClick={() => onOpenChange(false)}
                className="h-10 w-full rounded-lg"
              >
                সম্পন্ন
              </Button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="animate-fade-rise p-5 sm:p-6">
            {/* <div className="mb-5 rounded-lg border border-border bg-secondary/45 p-4">
              <div className="flex gap-3">
                <span className="inline-flex size-9 shrink-0 items-center justify-center rounded-lg bg-background text-primary">
                  <Send className="size-4" />
                </span>
                <div className="min-w-0">
                  <h2 className="font-heading text-base font-bold">মুফতি আমন্ত্রণ পাঠান</h2>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    আমন্ত্রিত মুফতি ইমেইল এবং পাসওয়ার্ড সেটআপ লিংক পাবেন
                  </p>
                </div>
              </div>
            </div> */}

            {error ? (
              <p className="mb-4 rounded-lg border border-destructive/25 bg-destructive/10 px-3.5 py-2.5 text-sm text-destructive">
                {error}
              </p>
            ) : null}

            <div className="space-y-4">
              <div className="space-y-1.5">
                <label htmlFor="Lawyer-invite-name" className="text-sm font-medium text-foreground">
                  পূর্ণ নাম
                </label>
                <div className="relative">
                  <UserRound className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    id="Lawyer-invite-name"
                    type="text"
                    value={form.name}
                    onChange={(event) => handleChange("name", event.target.value)}
                    disabled={formState === "submitting"}
                    placeholder="লইয়ারের পূর্ণ নাম"
                    autoComplete="off"
                    autoFocus
                    className="h-11 w-full rounded-lg border border-input bg-background px-10 text-sm outline-none transition placeholder:text-muted-foreground/70 disabled:cursor-not-allowed disabled:opacity-60 focus:border-ring focus:ring-2 focus:ring-ring/30"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label htmlFor="Lawyer-invite-email" className="text-sm font-medium text-foreground">
                  ইমেইল ঠিকানা
                </label>
                <div className="relative">
                  <Mail className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    id="Lawyer-invite-email"
                    type="email"
                    value={form.email}
                    onChange={(event) => handleChange("email", event.target.value)}
                    disabled={formState === "submitting"}
                    placeholder="Lawyer@example.com"
                    autoComplete="off"
                    className="h-11 w-full rounded-lg border border-input bg-background px-10 text-sm outline-none transition placeholder:text-muted-foreground/70 disabled:cursor-not-allowed disabled:opacity-60 focus:border-ring focus:ring-2 focus:ring-ring/30"
                  />
                </div>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={formState === "submitting"}
                className="h-11 w-full rounded-lg"
              >
                বাতিল
              </Button>
              <Button
                type="submit"
                disabled={formState === "submitting"}
                className="h-11 w-full rounded-lg text-base"
              >
                <span className="inline-flex size-4 items-center justify-center">
                  {formState === "submitting" ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Send className="size-4" />
                  )}
                </span>
                আমন্ত্রণ পাঠান
              </Button>
            </div>
          </form>
        )}
      </DialogContent>
    </Dialog>
  )
}
