import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { percent } from "@/lib/format"
import type { Source } from "@/lib/types"
import { FileText, Hash } from "lucide-react"

export function ThinkingModal({
  source,
  onClose,
}: {
  source: Source | null
  onClose: () => void
}) {
  return (
    <Dialog open={Boolean(source)} onOpenChange={(open) => (!open ? onClose() : null)}>
      <DialogContent className="max-h-[88vh] gap-0 overflow-hidden p-0 sm:max-w-2xl">
        {source ? (
          <>
            <DialogHeader className="space-y-2 border-b border-border p-5 text-left">
              <p className="text-xs font-semibold uppercase tracking-wide text-primary">
                মূল অংশ
              </p>
              <DialogTitle dir="auto" className="font-heading text-lg leading-snug">
                {source.law_name}
              </DialogTitle>
              <div className="flex flex-wrap gap-2 pt-1">
                <Badge variant="secondary" className="gap-1.5 font-normal">
                  <FileText className="size-3.5 text-primary" />
                  {source.link || "লিংক নেই"}
                </Badge>
                <Badge variant="secondary" className="gap-1.5 font-normal">
                  <Hash className="size-3.5 text-primary" />
                  {source.year || "বছর নেই"}
                </Badge>
                <Badge className="border-0 bg-accent text-accent-foreground">
                  {percent(source.relevance_score)} প্রাসঙ্গিক
                </Badge>
              </div>
            </DialogHeader>

            <div className="max-h-[55vh] overflow-y-auto p-5">
              <pre
                dir="auto"
                className="whitespace-pre-wrap rounded-xl border border-border bg-secondary/40 p-4 font-sans text-sm leading-loose text-foreground"
              >
                {source.context_text || "API থেকে কোনো অংশ পাওয়া যায়নি।"}
              </pre>
            </div>
          </>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
