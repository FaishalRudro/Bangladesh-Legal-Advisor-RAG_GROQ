import { useEffect, useRef, useState, type FormEvent } from "react"
import { BrandMark } from "@/components/brand-mark"
import { StreamingAnswer } from "@/components/streaming-answer"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { confidenceLabels, type DraftConversation, type ChatTurn, type ServerState } from "@/lib/app-types"
import { ArrowUp, BookOpen, Check, Copy, FileSearch, Mail, Menu, PanelRightClose, PanelRightOpen, ShieldQuestion, Sparkles, User, Settings, LogOut, ChevronDown, ThumbsUp, ThumbsDown } from "lucide-react"

import type { AuthSession } from "@/lib/types"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuGroup,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

const confidenceStyles: Record<string, string> = {
  high: "bg-accent text-accent-foreground",
  medium: "bg-secondary text-secondary-foreground",
  low: "bg-muted text-muted-foreground",
  not_found: "bg-muted text-muted-foreground",
}

const suggestions = [
  "পারিবারিক আইনে বিবাহ বিচ্ছেদের নিয়ম কী?",
  "জমি কেনার আগে কী কী বিষয় যাচাই করা উচিত?",
  "চেক ডিজঅনার হলে আইনি প্রতিকার কী?",
  "ভোক্তা অধিকার আইনে অভিযোগ করার নিয়ম কী?",
]

const serverLabel: Record<ServerState, string> = {
  checking: "সংযোগ যাচাই হচ্ছে",
  online: "সার্ভার সক্রিয়",
  offline: "সার্ভার বন্ধ",
}

const serverDot: Record<ServerState, string> = {
  checking: "bg-warning",
  online: "bg-primary",
  offline: "bg-destructive",
}

export function ChatMain({
  conversation,
  selectedTurnId,
  streamingTurnId,
  isAsking,
  pendingQuestion,
  chatError,
  query,
  serverState,
  onQueryChange,
  onSubmit,
  onSelectTurn,
  onStreamDone,
  onOpenMobileMenu,
  onOpenSources,
  session,
  onLogout,
  onFeedback,
  onOpenSettings,
  hasMoreMessages,
  isLoadingMoreMessages,
  onLoadMoreMessages,
}: {
  conversation: DraftConversation | undefined
  selectedTurnId: string | null
  streamingTurnId: string | null
  isAsking: boolean
  pendingQuestion: string
  chatError: string
  query: string
  serverState: ServerState
  onQueryChange: (value: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onSelectTurn: (id: string) => void
  onStreamDone: () => void
  onOpenMobileMenu: () => void
  onOpenSources?: () => void
  session: AuthSession
  onLogout: () => void
  onFeedback?: (messageId: string, isGood: boolean) => void
  onOpenSettings: (tab: "profile" | "security") => void
  hasMoreMessages?: boolean
  isLoadingMoreMessages?: boolean
  onLoadMoreMessages?: () => void
}) {
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const turns = conversation?.turns ?? []
  const hasMessages = turns.length > 0 || Boolean(pendingQuestion)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [pendingQuestion, turns.length, isAsking])

  useEffect(() => {
    adjustTextareaHeight()
  }, [query])

  function adjustTextareaHeight() {
    const textarea = textareaRef.current
    if (!textarea) return
    
    // Reset height to auto to get the correct scrollHeight
    textarea.style.height = "auto"
    // Set height to scrollHeight to fit content
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px"
  }

  function handleSuggestion(text: string) {
    onQueryChange(text)
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <section className="relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background">
      {/* Top bar */}
      <header className="flex h-[72px] shrink-0 items-center justify-between gap-3 border-b border-border bg-card/60 px-4 backdrop-blur md:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <button
            type="button"
            onClick={onOpenMobileMenu}
            className="inline-flex size-9 items-center justify-center rounded-lg border border-border text-muted-foreground lg:hidden"
            aria-label="মেনু খুলুন"
          >
            <Menu className="size-5" />
          </button>
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wide text-primary">Chat</p>
            <h1 className="truncate font-heading text-base font-bold leading-tight md:text-lg">
              {conversation?.title || "নতুন আলোচনা"}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="hidden items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground sm:inline-flex">
            <span className={cn("size-2 rounded-full", serverDot[serverState])} />
            {serverLabel[serverState]}
          </span>
          
          <DropdownMenu>
            <DropdownMenuTrigger
              className="flex items-center gap-2 rounded-full border border-border bg-card py-1.5 pl-1.5 pr-3 text-sm font-medium transition hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring cursor-pointer"
            >
              <span className="inline-flex size-6 items-center justify-center rounded-full bg-primary font-semibold text-primary-foreground text-xs">
                {session.user.name.slice(0, 1).toUpperCase()}
              </span>
              <span className="hidden md:inline-block max-w-[100px] truncate text-foreground">
                {session.user.name}
              </span>
              <ChevronDown className="size-3.5 text-muted-foreground" />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuGroup>
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-0.5">
                    <p className="text-sm font-semibold leading-none">{session.user.name}</p>
                    <p className="text-xs leading-none text-muted-foreground pt-1 flex items-center gap-1.5">
                      <Mail className="size-3" />
                      {session.user.email}
                    </p>
                  </div>
                </DropdownMenuLabel>
              </DropdownMenuGroup>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="cursor-pointer h-9" onClick={() => onOpenSettings("profile")}>
                <User className="mr-2.5 size-4" />
                <span>Profile</span>
              </DropdownMenuItem>
              <DropdownMenuItem className="cursor-pointer h-9" onClick={() => onOpenSettings("security")}>
                <Settings className="mr-2.5 size-4" />
                <span>Settings</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="cursor-pointer h-9 text-destructive focus:bg-destructive/10 focus:text-destructive" onClick={onLogout}>
                <LogOut className="mr-2.5 size-4" />
                <span>Logout</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-8 pt-6 md:px-6">
        {!hasMessages ? (
          <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center text-center">
            <BrandMark size={68} />
            <h2 className="mt-5 text-balance font-heading text-2xl font-bold md:text-3xl">
              কী জানতে চান?
            </h2>
            <p className="mt-2 text-pretty leading-relaxed text-muted-foreground">
              বাংলা বা ইংরেজিতে প্রশ্ন করুন। প্রতিটি উত্তরের সাথে নির্ভরযোগ্য
              তথ্যসূত্র দেখানো হবে।
            </p>

            <div className="mt-8 grid w-full gap-3 sm:grid-cols-2 max-w-2xl mx-auto">
              {suggestions.map((text) => (
                <button
                  key={text}
                  type="button"
                  onClick={() => handleSuggestion(text)}
                  className="group flex w-full items-center gap-2.5 rounded-full border border-border bg-card/60 px-5 py-3 text-sm font-medium text-foreground/80 transition-all duration-300 hover:border-primary/40 hover:bg-primary/5 hover:text-primary hover:shadow-sm"
                >
                  <Sparkles className="size-4 text-primary/60 transition-colors group-hover:text-primary" />
                  <span className="truncate max-w-[280px] sm:max-w-none">{text}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-10">
            {hasMoreMessages && (
              <div className="flex justify-center pb-4">
                <button
                  type="button"
                  onClick={onLoadMoreMessages}
                  disabled={isLoadingMoreMessages}
                  className="rounded-full border border-border bg-card px-4 py-2 text-xs font-medium text-muted-foreground shadow-sm transition hover:bg-accent hover:text-foreground disabled:opacity-50"
                >
                  {isLoadingMoreMessages ? "Loading previous..." : "Load previous messages"}
                </button>
              </div>
            )}
            
            {turns.map((turn) => (
              <TurnBlock
                key={turn.id}
                turn={turn}
                isSelected={selectedTurnId === turn.id}
                isStreaming={streamingTurnId === turn.id}
                onSelect={() => onSelectTurn(turn.id)}
                onStreamDone={onStreamDone}
                isLawyer={session?.user?.role === "Lawyer" || session?.user?.role === "super_admin"}
                onFeedback={onFeedback}
              />
            ))}

            {pendingQuestion ? <UserQuestionBubble question={pendingQuestion} /> : null}

            {isAsking ? (
              <div className="animate-fade-rise flex items-start gap-3">
                <BrandMark size={36} />
                <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm border border-border bg-card px-4 py-3 text-sm text-muted-foreground shadow-sm">
                  চিন্তা করছি
                  <span className="flex items-center gap-1" aria-hidden="true">
                    <span className="thinking-dot" />
                    <span className="thinking-dot" />
                    <span className="thinking-dot" />
                  </span>
                </div>
              </div>
            ) : null}
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="sticky bottom-0 z-20 shrink-0 bg-gradient-to-b from-background to-card/30 px-4 pb-[calc(1rem+env(safe-area-inset-bottom))] pt-4 md:px-6">
        <form onSubmit={onSubmit} className="mx-auto max-w-3xl">
          {chatError ? (
            <p className="mb-3 rounded-xl border border-destructive/30 bg-destructive/10 px-3.5 py-2.5 text-sm text-destructive">
              {chatError}
            </p>
          ) : null}

          <div className="flex items-end gap-2 rounded-xl border border-border bg-card p-2 shadow-sm transition focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/20">
            <textarea
              ref={textareaRef}
              dir="auto"
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="আপনার প্রশ্ন লিখুন..."
              className="min-h-10 max-h-40 min-w-0 flex-1 resize-none overflow-y-auto bg-transparent px-2.5 py-2 text-sm leading-relaxed outline-none placeholder:text-muted-foreground"
            />
            <button
              type="submit"
              disabled={isAsking || serverState === "offline" || query.trim().length < 5}
              className="inline-flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
              aria-label="প্রশ্ন পাঠান"
            >
              <ArrowUp className="size-5" />
            </button>
          </div>
          <p className="mt-2 px-1 text-center text-xs text-muted-foreground">
            Legal Advice GPT ভুল করতে পারে — গুরুত্বপূর্ণ বিষয়ে আলেমের পরামর্শ নিন।
          </p>
        </form>
      </div>
    </section>
  )
}

function UserQuestionBubble({ question }: { question: string }) {
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    navigator.clipboard.writeText(question)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="flex justify-end group">
      <div className="flex flex-col items-end gap-1 max-w-[85%] sm:max-w-[75%]">
        <div
          dir="auto"
          className="rounded-3xl rounded-tr-sm bg-secondary px-5 py-3 text-base leading-relaxed text-foreground text-justify"
        >
          {question}
        </div>
        <button
          onClick={handleCopy}
          className="inline-flex items-center gap-1.5 px-2 py-1 text-[11px] font-medium text-muted-foreground opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100"
          title="প্রশ্ন কপি করুন"
        >
          {copied ? (
            <>
              <Check className="size-3" /> কপি হয়েছে
            </>
          ) : (
            <>
              <Copy className="size-3" /> কপি করুন
            </>
          )}
        </button>
      </div>
    </div>
  )
}

function TurnBlock({
  turn,
  isSelected,
  isStreaming,
  onSelect,
  onStreamDone,
  isLawyer,
  onFeedback,
}: {
  turn: ChatTurn
  isSelected: boolean
  isStreaming: boolean
  onSelect: () => void
  onStreamDone: () => void
  isLawyer?: boolean
  onFeedback?: (messageId: string, isGood: boolean) => void
}) {
  const [copied, setCopied] = useState(false)

  function handleCopyAnswer() {
    navigator.clipboard.writeText(turn.response.answer)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const userFeedback = turn.feedbacks?.find((f) => f.lawyer_name) // find my feedback, or just check general feedback logic
  const isThumbsUp = userFeedback?.is_good === true;
  const isThumbsDown = userFeedback?.is_good === false;

  return (
    <article className="animate-fade-rise space-y-6">
      {/* User question */}
      <UserQuestionBubble question={turn.question} />

      {/* Assistant answer */}
      <div className="flex items-start gap-4">
        <div className="mt-1 shrink-0">
          <BrandMark size={32} />
        </div>
        <div className="min-w-0 flex-1 space-y-4">
          <StreamingAnswer
            shouldStream={isStreaming}
            text={turn.response.answer}
            onDone={onStreamDone}
          />

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleCopyAnswer}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-semibold text-muted-foreground transition hover:bg-accent/40"
              title="উত্তর কপি করুন"
            >
              {copied ? (
                <>
                  <Check className="size-3.5" />
                  কপি হয়েছে
                </>
              ) : (
                <>
                  <Copy className="size-3.5" />
                  কপি করুন
                </>
              )}
            </button>
            <Badge
              className={cn(
                "gap-1.5 border-0 text-xs font-medium",
                confidenceStyles[turn.response.confidence],
              )}
            >
              <ShieldQuestion className="size-3.5" />
              {confidenceLabels[turn.response.confidence]}
            </Badge>

            <button
              type="button"
              onClick={onSelect}
              className={cn(
                "inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm font-semibold transition",
                isSelected
                  ? "border-primary/40 bg-accent/60 text-primary shadow-sm hover:bg-accent"
                  : "border-border bg-card text-muted-foreground hover:bg-accent/40",
              )}
            >
              <FileSearch className="size-4" />
              তথ্যসূত্র দেখুন ({turn.response.sources?.length ?? 0})
            </button>
            {isLawyer && onFeedback && !turn.id.startsWith("turn") && (
              <div className="flex items-center gap-1 border-l border-border pl-3 ml-1">
                <button
                  type="button"
                  onClick={() => onFeedback(turn.id, true)}
                  className={cn(
                    "inline-flex size-8 items-center justify-center rounded-lg border border-border bg-card transition hover:bg-accent/40",
                    isThumbsUp && "text-green-600 border-green-600/40 bg-green-50"
                  )}
                  title="সঠিক উত্তর"
                >
                  <ThumbsUp className="size-4" />
                </button>
                <button
                  type="button"
                  onClick={() => onFeedback(turn.id, false)}
                  className={cn(
                    "inline-flex size-8 items-center justify-center rounded-lg border border-border bg-card transition hover:bg-accent/40",
                    isThumbsDown && "text-red-600 border-red-600/40 bg-red-50"
                  )}
                  title="ভুল উত্তর"
                >
                  <ThumbsDown className="size-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </article>
  )
}
