import { Badge } from "@/components/ui/badge"
import { percent } from "@/lib/format"
import { type ChatTurn, relevanceLabels } from "@/lib/app-types"
import type { Source } from "@/lib/types"
import { FileText, Languages, Layers, Library, ScrollText, Sparkles, X } from "lucide-react"

const relevanceStyles: Record<"high" | "medium" | "low", string> = {
  high: "bg-accent text-accent-foreground",
  medium: "bg-secondary text-secondary-foreground",
  low: "bg-muted text-muted-foreground",
}

export function SourcesPanel({
  selectedTurn,
  onViewThinking,
  onClose,
}: {
  selectedTurn: ChatTurn | null
  onViewThinking: (source: Source) => void
  onClose?: () => void
}) {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-card">
      <header className="flex h-[72px] shrink-0 items-center justify-between gap-2.5 border-b border-border px-5">
        <div className="flex items-center gap-2.5">
          <span className="inline-flex size-8 items-center justify-center rounded-lg bg-accent text-primary">
            <Library className="size-5" />
          </span>
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-primary">
              Evidence
            </p>
            <h2 className="font-heading text-base font-bold leading-tight">তথ্যসূত্র</h2>
          </div>
        </div>
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            className="inline-flex size-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="বন্ধ করুন"
          >
            <X className="size-5" />
          </button>
        ) : null}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-4">
        {selectedTurn ? (
          <div className="space-y-4">
            {/* Context summary */}
            <div className="rounded-xl border border-border bg-secondary/50 p-3.5">
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary" className="gap-1.5 bg-card font-normal">
                  <Layers className="size-3.5 text-primary" />
                  {selectedTurn.response.total_chunks_retrieved} chunks
                </Badge>
                <Badge variant="secondary" className="gap-1.5 bg-card font-normal">
                  <Languages className="size-3.5 text-primary" />
                  {selectedTurn.response.query_language === "bn" ? "বাংলা" : "English"}
                </Badge>
              </div>
              {selectedTurn.response.legal_query ? (
                <p
                  className="mt-3 border-t border-border pt-3 text-sm text-muted-foreground"
                >
                  <span className="font-semibold text-foreground">Legal Query Formulated:</span> {selectedTurn.response.legal_query}
                </p>
              ) : null}
            </div>

            {/* Sources */}
            <div className="space-y-3">
              {selectedTurn.response.sources.map((source, index) => (
                <article
                  key={`${source.link}-${index}`}
                  className="rounded-xl border border-border bg-card p-3.5 transition hover:border-primary/40 hover:shadow-sm"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex min-w-0 flex-1 items-start gap-2">
                      <span className="mt-0.5 inline-flex size-7 shrink-0 items-center justify-center rounded-md bg-accent text-primary">
                        <ScrollText className="size-4" />
                      </span>
                      <strong
                        dir="auto"
                        className="text-sm font-semibold leading-snug text-foreground"
                      >
                        {source.law_name}
                      </strong>
                    </div>
                    {source.year ? (
                      <span className="shrink-0 text-xs font-medium text-muted-foreground whitespace-nowrap mt-0.5">
                        বছর: {source.year}
                      </span>
                    ) : null}
                  </div>
                  
                  {source.repealed && source.repealed !== "ACTIVE" && (
                     <div className="mt-2 text-xs font-semibold text-destructive">
                       [রহিতকৃত / REPEALED]
                     </div>
                  )}

                  <button
                    type="button"
                    onClick={() => onViewThinking(source)}
                    className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-primary/40 bg-accent/40 py-2 text-sm font-semibold text-primary transition hover:bg-accent"
                  >
                    <FileText className="size-4" />
                    মূল অংশ দেখুন
                  </button>
                </article>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center px-6 text-center">
            <span className="inline-flex size-12 items-center justify-center rounded-2xl bg-accent text-primary">
              <Sparkles className="size-6" />
            </span>
            <p className="mt-4 text-sm font-medium text-foreground">
              এখনো কোনো তথ্যসূত্র নেই
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              প্রশ্ন করলে এখানে প্রাসঙ্গিক উৎস ও রেফারেন্স দেখানো হবে।
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
