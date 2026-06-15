import { useState, useEffect } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { usersApi } from "@/lib/api/users"
import type { AuthSession } from "@/lib/types"
import { useAuthContext } from "@/lib/auth-context"
import { getApiErrorMessage } from "@/lib/http"
import { cn } from "@/lib/utils"
import { UserCog, ShieldCheck, Mail, User, Tag, Lock, KeyRound, Eye, EyeOff } from "lucide-react"

export function SettingsDialog({
  open,
  onOpenChange,
  session,
  defaultTab = "profile"
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  session: AuthSession
  defaultTab?: "profile" | "security"
}) {
  const { fetchSession } = useAuthContext()
  const [tab, setTab] = useState(defaultTab)
  
  useEffect(() => {
    if (open) {
      setTab(defaultTab)
    }
  }, [open, defaultTab])

  // Profile Form
  const [name, setName] = useState(session.user.name)
  const [isUpdatingProfile, setIsUpdatingProfile] = useState(false)
  const [profileMessage, setProfileMessage] = useState("")
  const [profileError, setProfileError] = useState("")

  // Security Form
  const [oldPassword, setOldPassword] = useState("")
  const [showOldPassword, setShowOldPassword] = useState(false)
  const [newPassword, setNewPassword] = useState("")
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [otp, setOtp] = useState("")
  const [isChangingPassword, setIsChangingPassword] = useState(false)
  const [passwordStep, setPasswordStep] = useState<"request" | "verify">("request")
  const [securityMessage, setSecurityMessage] = useState("")
  const [securityError, setSecurityError] = useState("")

  async function handleUpdateProfile(e: React.FormEvent) {
    e.preventDefault()
    setIsUpdatingProfile(true)
    setProfileMessage("")
    setProfileError("")
    try {
      await usersApi.updateProfile(name)
      await fetchSession()
      setProfileMessage("প্রোফাইল সফলভাবে আপডেট হয়েছে।")
    } catch (err) {
      setProfileError(getApiErrorMessage(err) || "প্রোফাইল আপডেট করতে ব্যর্থ হয়েছে।")
    } finally {
      setIsUpdatingProfile(false)
    }
  }

  async function handleRequestPasswordChange(e: React.FormEvent) {
    e.preventDefault()
    setIsChangingPassword(true)
    setSecurityMessage("")
    setSecurityError("")
    try {
      const res = await usersApi.requestPasswordChange(oldPassword)
      setPasswordStep("verify")
      setSecurityMessage(res.detail || "আপনার ইমেইলে একটি ওটিপি (OTP) পাঠানো হয়েছে।")
    } catch (err) {
      setSecurityError(getApiErrorMessage(err) || "পাসওয়ার্ড পরিবর্তনের অনুরোধ ব্যর্থ হয়েছে।")
    } finally {
      setIsChangingPassword(false)
    }
  }

  async function handleVerifyPasswordChange(e: React.FormEvent) {
    e.preventDefault()
    setIsChangingPassword(true)
    setSecurityMessage("")
    setSecurityError("")
    try {
      const res = await usersApi.verifyPasswordChange(otp, newPassword)
      setSecurityMessage(res.detail || "পাসওয়ার্ড সফলভাবে পরিবর্তন করা হয়েছে।")
      setPasswordStep("request")
      setOldPassword("")
      setNewPassword("")
      setOtp("")
    } catch (err) {
      setSecurityError(getApiErrorMessage(err) || "পাসওয়ার্ড পরিবর্তন করতে ব্যর্থ হয়েছে।")
    } finally {
      setIsChangingPassword(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[550px] p-0 overflow-hidden gap-0">
        <DialogHeader className="p-6 sm:p-8 pb-6 border-b bg-gradient-to-br from-primary/5 to-transparent">
          <DialogTitle className="font-heading text-2xl font-bold text-primary">সেটিংস</DialogTitle>
          <DialogDescription className="text-base text-foreground/80 mt-1.5 font-medium">
            আপনার অ্যাকাউন্টের প্রোফাইল এবং নিরাপত্তা পরিচালনা করুন
          </DialogDescription>
        </DialogHeader>

        <div className="px-6 sm:px-8 py-6">
          <Tabs value={tab} onValueChange={(v) => setTab(v as any)} className="w-full">
            <TabsList className="grid w-full grid-cols-2 rounded-lg h-10 bg-muted p-0.5 gap-0.5">
              <TabsTrigger value="profile" className="rounded-md text-sm font-bold h-full flex items-center justify-center px-3">
                <UserCog className="mr-1.5 size-4" />
                প্রোফাইল
              </TabsTrigger>
              <TabsTrigger value="security" className="rounded-md text-sm font-bold h-full flex items-center justify-center px-3">
                <ShieldCheck className="mr-1.5 size-4" />
                নিরাপত্তা
              </TabsTrigger>
            </TabsList>

            <div className="mt-6 pb-2 min-h-[360px]">
              <TabsContent value="profile" className="mt-0 outline-none space-y-5 animate-fade-rise">
                <form onSubmit={handleUpdateProfile} className="space-y-5">
                  <div className="space-y-5">
                    <div className="space-y-2.5">
                      <label htmlFor="email" className="text-sm font-medium text-foreground flex items-center gap-2">
                        <Mail className="size-4 text-muted-foreground" />
                        ইমেইল এড্রেস
                      </label>
                      <Input 
                        id="email" 
                        value={session.user.email} 
                        disabled 
                        className="h-11 text-sm" 
                      />
                    </div>
                    
                    <div className="space-y-2.5">
                      <label htmlFor="role" className="text-sm font-medium text-foreground flex items-center gap-2">
                        <Tag className="size-4 text-muted-foreground" />
                        রোল (Role)
                      </label>
                      <Input 
                        id="role" 
                        value={session.user.role} 
                        disabled 
                        className="h-11 text-sm capitalize" 
                      />
                    </div>

                    <div className="space-y-2.5">
                      <label htmlFor="name" className="text-sm font-medium text-foreground flex items-center gap-2">
                        <User className="size-4 text-muted-foreground" />
                        আপনার নাম
                      </label>
                      <Input
                        id="name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="আপনার নাম লিখুন"
                        required
                        className="h-11 text-sm"
                      />
                    </div>
                  </div>

                  {profileError && (
                    <div className="rounded-lg border-2 border-red-400 bg-red-50 dark:bg-red-950/40 px-4 py-3.5 text-sm font-semibold text-red-700 dark:text-red-400 flex items-start gap-2.5">
                      <span className="text-xl leading-none">⚠️</span>
                      <span className="flex-1">{profileError}</span>
                    </div>
                  )}
                  {profileMessage && (
                    <div className="rounded-lg border-2 border-emerald-400 bg-emerald-50 dark:bg-emerald-950/40 px-4 py-3.5 text-sm font-semibold text-emerald-700 dark:text-emerald-400 flex items-start gap-2.5">
                      <span className="text-xl leading-none">✓</span>
                      <span className="flex-1">{profileMessage}</span>
                    </div>
                  )}

                  <Button 
                    type="submit" 
                    className="h-12 text-base w-full rounded-lg font-bold shadow-md hover:shadow-lg transition-all bg-primary hover:bg-primary/90 text-primary-foreground" 
                    disabled={isUpdatingProfile || name === session.user.name}
                  >
                    {isUpdatingProfile ? "আপডেট হচ্ছে..." : "প্রোফাইল আপডেট করুন"}
                  </Button>
                </form>
              </TabsContent>

              <TabsContent value="security" className="mt-0 outline-none space-y-5 animate-fade-rise">
                <div className="rounded-lg border-2 border-primary/30 bg-gradient-to-br from-primary/10 to-primary/5 p-4">
                  <div className="flex gap-3">
                    <span className="inline-flex size-11 shrink-0 items-center justify-center rounded-lg bg-primary/20 border-2 border-primary/40 text-primary shadow-sm">
                      <ShieldCheck className="size-5.5" />
                    </span>
                    <div className="min-w-0">
                      <h2 className="font-heading text-base font-bold text-foreground">পাসওয়ার্ড পরিবর্তন</h2>
                      <p className="mt-1.5 text-sm leading-relaxed text-foreground/75 font-medium">
                        {passwordStep === "request" 
                          ? "আপনার বর্তমান পাসওয়ার্ড দিয়ে নতুন পাসওয়ার্ডের জন্য ওটিপি (OTP) অনুরোধ করুন।"
                          : "ইমেইলে প্রাপ্ত ওটিপি (OTP) এবং নতুন পাসওয়ার্ড প্রদান করুন।"}
                      </p>
                    </div>
                  </div>
                </div>

                {passwordStep === "request" ? (
                  <form onSubmit={handleRequestPasswordChange} className="space-y-5">
                    <div className="space-y-2.5">
                      <label htmlFor="old_password" className="text-sm font-medium text-foreground flex items-center gap-2">
                        <Lock className="size-4 text-muted-foreground" />
                        বর্তমান পাসওয়ার্ড
                      </label>
                      <div className="relative">
                        <Input
                          id="old_password"
                          type={showOldPassword ? "text" : "password"}
                          value={oldPassword}
                          onChange={(e) => setOldPassword(e.target.value)}
                          placeholder="আপনার বর্তমান পাসওয়ার্ড"
                          required
                          className="h-11 text-sm pr-11"
                        />
                        <button
                          type="button"
                          onClick={() => setShowOldPassword(!showOldPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-primary/60 hover:text-primary transition-colors p-1 rounded-md hover:bg-primary/10"
                          aria-label={showOldPassword ? "Hide password" : "Show password"}
                        >
                          {showOldPassword ? <EyeOff className="size-4.5" /> : <Eye className="size-4.5" />}
                        </button>
                      </div>
                    </div>

                    {securityError && (
                      <div className="rounded-lg border-2 border-red-400 bg-red-50 dark:bg-red-950/40 px-4 py-3.5 text-sm font-semibold text-red-700 dark:text-red-400 flex items-start gap-2.5">
                        <span className="text-xl leading-none">⚠️</span>
                        <span className="flex-1">{securityError}</span>
                      </div>
                    )}
                    {securityMessage && (
                      <div className="rounded-lg border-2 border-emerald-400 bg-emerald-50 dark:bg-emerald-950/40 px-4 py-3.5 text-sm font-semibold text-emerald-700 dark:text-emerald-400 flex items-start gap-2.5">
                        <span className="text-xl leading-none">✓</span>
                        <span className="flex-1">{securityMessage}</span>
                      </div>
                    )}

                    <Button 
                      type="submit" 
                      className="h-12 text-base w-full rounded-lg font-bold shadow-md hover:shadow-lg transition-all bg-primary hover:bg-primary/90 text-primary-foreground" 
                      disabled={isChangingPassword || !oldPassword}
                    >
                      {isChangingPassword ? "ওটিপি (OTP) পাঠানো হচ্ছে..." : "পাসওয়ার্ড পরিবর্তন শুরু করুন"}
                    </Button>
                  </form>
                ) : (
                  <form onSubmit={handleVerifyPasswordChange} className="space-y-5">
                    <div className="space-y-2.5">
                      <label htmlFor="otp" className="text-sm font-medium text-foreground flex items-center gap-2">
                        <KeyRound className="size-4 text-muted-foreground" />
                        ওটিপি (OTP)
                      </label>
                      <Input
                        id="otp"
                        value={otp}
                        onChange={(e) => setOtp(e.target.value)}
                        placeholder="ইমেইলে প্রাপ্ত ওটিপি"
                        required
                        className="h-11 text-sm"
                      />
                    </div>
                    
                    <div className="space-y-2.5">
                      <label htmlFor="new_password" className="text-sm font-medium text-foreground flex items-center gap-2">
                        <Lock className="size-4 text-muted-foreground" />
                        নতুন পাসওয়ার্ড
                      </label>
                      <div className="relative">
                        <Input
                          id="new_password"
                          type={showNewPassword ? "text" : "password"}
                          value={newPassword}
                          onChange={(e) => setNewPassword(e.target.value)}
                          placeholder="নতুন শক্তিশালী পাসওয়ার্ড"
                          required
                          className="h-11 text-sm pr-11"
                        />
                        <button
                          type="button"
                          onClick={() => setShowNewPassword(!showNewPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-primary/60 hover:text-primary transition-colors p-1 rounded-md hover:bg-primary/10"
                          aria-label={showNewPassword ? "Hide password" : "Show password"}
                        >
                          {showNewPassword ? <EyeOff className="size-4.5" /> : <Eye className="size-4.5" />}
                        </button>
                      </div>
                    </div>

                    {securityError && (
                      <div className="rounded-lg border-2 border-red-400 bg-red-50 dark:bg-red-950/40 px-4 py-3.5 text-sm font-semibold text-red-700 dark:text-red-400 flex items-start gap-2.5">
                        <span className="text-xl leading-none">⚠️</span>
                        <span className="flex-1">{securityError}</span>
                      </div>
                    )}
                    {securityMessage && (
                      <div className="rounded-lg border-2 border-emerald-400 bg-emerald-50 dark:bg-emerald-950/40 px-4 py-3.5 text-sm font-semibold text-emerald-700 dark:text-emerald-400 flex items-start gap-2.5">
                        <span className="text-xl leading-none">✓</span>
                        <span className="flex-1">{securityMessage}</span>
                      </div>
                    )}

                    <Button 
                      type="submit" 
                      className="h-12 text-base w-full rounded-lg font-bold shadow-md hover:shadow-lg transition-all bg-primary hover:bg-primary/90 text-primary-foreground" 
                      disabled={isChangingPassword || !otp || !newPassword}
                    >
                      {isChangingPassword ? "যাচাই করা হচ্ছে..." : "নতুন পাসওয়ার্ড নিশ্চিত করুন"}
                    </Button>
                  </form>
                )}
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  )
}
