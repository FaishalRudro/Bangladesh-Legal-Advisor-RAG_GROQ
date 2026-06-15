import { Loader2 } from "lucide-react"

export function PageLoader() {
  return (
    <main className="grid min-h-screen place-items-center bg-background">
      <div className="flex flex-col items-center gap-4 text-muted-foreground animate-in fade-in duration-500">
        <Loader2 className="size-10 animate-spin text-primary" />
        <p className="text-sm font-medium tracking-wide">লোড হচ্ছে...</p>
      </div>
    </main>
  )
}
