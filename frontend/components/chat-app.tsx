import { type ChangeEvent, type FormEvent, useEffect, useMemo, useState } from "react"
import { useRouter } from "next/router"
import { useLaws } from "@/hooks/use-laws"
import { useChat } from "@/hooks/use-chat"
import { useHealth } from "@/hooks/use-health"
import { useAuthContext } from "@/lib/auth-context"
import { useSessions } from "@/hooks/use-sessions"
import { sessionsApi } from "@/lib/api/sessions"
import type { DraftConversation, ViewMode } from "@/lib/app-types"
import { createId } from "@/lib/format"
import type { LawInfo, Source } from "@/lib/types"
import { ChatSidebar } from "@/components/chat-sidebar"
import { ChatMain } from "@/components/chat-main"
import { SourcesPanel } from "@/components/sources-panel"
import { LawManager } from "@/components/law-manager"
import { SettingsDialog } from "@/components/settings-dialog"
import { ConfirmDialog } from "@/components/confirm-dialog"
import { CreateLawyerDialog } from "@/components/create-lawyer-panel"
import { ThinkingModal } from "@/components/thinking-modal"
import { Sheet, SheetContent } from "@/components/ui/sheet"
import { cn } from "@/lib/utils"

const initialConversations: DraftConversation[] = [
  {
    id: "local-new",
    title: "নতুন আলোচনা",
    turns: [],
    createdAt: new Date().toISOString(),
  },
]

export function ChatApp() {
  const { session, logout } = useAuthContext()
  const router = useRouter()

  const canManageLaws = session?.user.role === "super_admin"
  const canViewLaws = session?.user.role === "super_admin" || session?.user.role === "Lawyer"
  const health = useHealth(Boolean(session))
  const chat = useChat()
  const lawState = useLaws(Boolean(session && canViewLaws))
  const sessionState = useSessions(Boolean(session))

  const [conversations, setConversations] = useState<DraftConversation[]>(initialConversations)
  const [activeConversationId, setActiveConversationId] = useState(initialConversations[0].id)
  const [query, setQuery] = useState("")
  const [pendingTurn, setPendingTurn] = useState<{ conversationId: string; question: string } | null>(null)
  const [viewMode, setViewModeState] = useState<ViewMode>(() => {
    if (typeof window !== "undefined") {
      const urlParams = new URLSearchParams(window.location.search)
      const view = urlParams.get("view") as ViewMode
      if (view === "laws" || view === "chat") return view
      
      const saved = sessionStorage.getItem("viewMode") as ViewMode
      if (saved === "laws" || saved === "chat") return saved
    }
    return "chat"
  })

  function setViewMode(mode: ViewMode) {
    setViewModeState(mode)
    if (typeof window !== "undefined") {
      sessionStorage.setItem("viewMode", mode)
      const url = new URL(window.location.href)
      if (mode === "chat") {
        url.searchParams.delete("view")
      } else {
        url.searchParams.set("view", mode)
      }
      window.history.replaceState({}, "", url)
    }
  }
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null)
  const [thinkingSource, setThinkingSource] = useState<Source | null>(null)
  const [streamingTurnId, setStreamingTurnId] = useState<string | null>(null)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [mobileSourcesOpen, setMobileSourcesOpen] = useState(false)
  const [desktopSourcesOpen, setDesktopSourcesOpen] = useState(false)
  const [isCreateLawyerOpen, setIsCreateLawyerOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [settingsTab, setSettingsTab] = useState<"profile" | "security">("profile")
  
  const [messagePages, setMessagePages] = useState<Record<string, { page: number; hasMore: boolean }>>({})
  const [isLoadingMoreMessages, setIsLoadingMoreMessages] = useState(false)

  const [lawToDelete, setLawToDelete] = useState<LawInfo | null>(null)
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeConversationId) ?? conversations[0],
    [activeConversationId, conversations],
  )

  useEffect(() => {
    const savedChat = sessionStorage.getItem("activeChat")
    if (savedChat) {
      if (savedChat.startsWith("chat") || savedChat.startsWith("local")) {
        sessionStorage.setItem("activeChat", "local-new")
        setActiveConversationId("local-new")
      } else {
        setTimeout(() => {
          handleSelectConversation(savedChat, false)
        }, 0)
      }
    }
  }, [])

  useEffect(() => {
    setConversations((current) => {
      const serverIds = new Set(sessionState.sessions.map(s => String(s.id)))
      
      const localConversations = current.filter((c) => {
        const isLocal = c.id.startsWith("chat") || c.id === "local-new"
        // Keep local if it has turns or is active, AND is not already in server sessions
        return isLocal && !serverIds.has(c.id) && (c.id === activeConversationId || c.turns.length > 0)
      })
      
      const serverConversations = sessionState.sessions.map((s) => {
        const existing = current.find((c) => c.id === String(s.id))
        return existing
          ? { ...existing, title: s.title, isPinned: s.is_pinned }
          : { id: String(s.id), title: s.title, turns: [], createdAt: s.created_at, isPinned: s.is_pinned }
      })
      return [...localConversations, ...serverConversations]
    })
  }, [sessionState.sessions])

  const selectedTurn = useMemo(() => {
    const allTurns = activeConversation?.turns ?? []
    if (selectedTurnId) {
      return allTurns.find((t) => t.id === selectedTurnId) ?? allTurns.at(-1) ?? null
    }
    return allTurns.at(-1) ?? null
  }, [activeConversation, selectedTurnId])

  function handleLogout() {
    logout()
    router.replace("/login")
  }

  function handleNewChat() {
    const conversation: DraftConversation = {
      id: createId("chat"),
      title: "নতুন আলোচনা",
      turns: [],
      createdAt: new Date().toISOString(),
    }
    setConversations((current) => [conversation, ...current])
    setActiveConversationId(conversation.id)
    sessionStorage.setItem("activeChat", conversation.id)
    setSelectedTurnId(null)
    setThinkingSource(null)
    setPendingTurn(null)
    setViewMode("chat")
    setMobileNavOpen(false)
  }

  async function handleSelectConversation(id: string, switchView = true) {
    setActiveConversationId(id)
    sessionStorage.setItem("activeChat", id)
    if (switchView) {
      setViewMode("chat")
    }
    setMobileNavOpen(false)

    const conversation = conversations.find((c) => c.id === id)
    if (!id.startsWith("local") && !id.startsWith("chat") && (!conversation || conversation.turns.length === 0)) {
      try {
        const data = await sessionsApi.getSession(Number(id))
        const turns = data.messages.map((m, i) => {
          // If the backend returns ai_response (as object or JSON string), extract the ChatResponse fields from it
          let parsedResponse = m
          if ('ai_response' in m && m.ai_response) {
            try {
              parsedResponse = typeof m.ai_response === 'string' ? JSON.parse(m.ai_response) : m.ai_response
            } catch (e) {
              console.error("Failed to parse ai_response", e)
            }
          }

          return {
            id: String(m.message_id ?? m.id ?? `turn-${i}`),
            question: m.user_query || m.query || "",
            response: {
              ...m,
              ...parsedResponse
            },
            createdAt: m.created_at,
            feedbacks: m.feedbacks,
          }
        })
        setConversations(cur => {
          const exists = cur.some(c => c.id === id)
          if (exists) {
            return cur.map(c => c.id === id ? { ...c, turns, title: data.title } : c)
          } else {
            return [...cur, { id, title: data.title, turns, createdAt: data.created_at, isPinned: data.is_pinned }]
          }
        })
        setMessagePages(prev => ({
          ...prev,
          [id]: { page: data.page, hasMore: data.page * data.size < data.total_messages }
        }))
        setSelectedTurnId(turns.at(-1)?.id ?? null)
      } catch (err) {
        console.error(err)
      }
    } else {
      setSelectedTurnId(conversation?.turns.at(-1)?.id ?? null)
    }
    setThinkingSource(null)
  }

  function handleOpenLaws() {
    setViewMode("laws")
    setMobileNavOpen(false)
  }

  function handleOpenLawyerManagement() {
    setMobileNavOpen(false)
    setIsCreateLawyerOpen(true)
  }

  async function handleLoadMoreMessages() {
    if (!activeConversationId || activeConversationId.startsWith("local") || activeConversationId.startsWith("chat")) return
    
    const pageData = messagePages[activeConversationId]
    if (!pageData || !pageData.hasMore) return

    setIsLoadingMoreMessages(true)
    try {
      const data = await sessionsApi.getSession(Number(activeConversationId), pageData.page + 1)
      const newTurns = data.messages.map((m, i) => {
        let parsedResponse = m
        if ('ai_response' in m && m.ai_response) {
          try {
            parsedResponse = typeof m.ai_response === 'string' ? JSON.parse(m.ai_response) : m.ai_response
          } catch (e) {
            console.error("Failed to parse ai_response", e)
          }
        }
        return {
          id: String(m.message_id ?? m.id ?? `turn-${i}`),
          question: m.user_query || m.query || "",
          response: { ...m, ...parsedResponse },
          createdAt: m.created_at,
          feedbacks: m.feedbacks,
        }
      })
      
      setConversations(cur => cur.map(c => 
        c.id === activeConversationId ? { ...c, turns: [...newTurns, ...c.turns] } : c
      ))
      setMessagePages(prev => ({
        ...prev,
        [activeConversationId]: { page: data.page, hasMore: data.page * data.size < data.total_messages }
      }))
    } catch (err) {
      console.error(err)
    } finally {
      setIsLoadingMoreMessages(false)
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    chat.setChatError("")
    setThinkingSource(null)

    const trimmedQuery = query.trim()
    if (trimmedQuery.length < 5) {
      chat.setChatError("Question must be at least 5 characters.")
      return
    }

    const targetConversationId = activeConversationId
    setPendingTurn({ conversationId: targetConversationId, question: trimmedQuery })
    setQuery("")

    try {
      const isLocal = targetConversationId.startsWith("local") || targetConversationId.startsWith("chat");
      const response = await chat.ask({
        query: trimmedQuery,
        law_id: null,
        top_k: 5,
        session_id: isLocal ? undefined : Number(targetConversationId),
      })
      const turn = {
        id: createId("turn"),
        question: trimmedQuery,
        response,
        createdAt: new Date().toISOString(),
      }

      const newConversationId = response.session_id ? String(response.session_id) : targetConversationId

      setConversations((current) =>
        current.map((conversation) => {
          if (conversation.id !== targetConversationId) return conversation
          return {
            ...conversation,
            id: newConversationId,
            title: conversation.turns.length === 0 ? trimmedQuery.slice(0, 44) : conversation.title,
            turns: [...conversation.turns, turn],
          }
        }),
      )
      
      if (newConversationId !== targetConversationId) {
        setActiveConversationId(newConversationId)
        sessionStorage.setItem("activeChat", newConversationId)
      }
      
      setSelectedTurnId(turn.id)
      setStreamingTurnId(turn.id)
      
      if (isLocal) {
        sessionState.loadSessions()
      }
    } catch {
      setQuery(trimmedQuery)
    } finally {
      setPendingTurn(null)
    }
  }

  async function handleFeedback(messageId: string, isGood: boolean) {
    try {
      await sessionsApi.submitFeedback(Number(messageId), isGood)
      // Update local state to reflect the feedback instantly if needed,
      // or rely on reload. For now, let's just do a optimistic update:
      setConversations(current => current.map(c => ({
        ...c,
        turns: c.turns.map(t => {
          if (t.id === messageId) {
            const feedbacks = [...(t.feedbacks || [])]
            const existingIdx = feedbacks.findIndex(f => f.lawyer_name === session?.user.name)
            if (existingIdx >= 0) {
              feedbacks[existingIdx].is_good = isGood
            } else {
              feedbacks.push({ is_good: isGood, feedback_text: null, lawyer_name: session?.user.name })
            }
            return { ...t, feedbacks }
          }
          return t
        })
      })))
    } catch (e) {
      console.error(e)
    }
  }

  function handleIngest(limit: number) {
    lawState.ingestLaws(limit)
  }

  function handleDelete(law: LawInfo) {
    if (!canManageLaws) {
      lawState.setUploadMessage("This account does not have permission to manage laws.")
      return
    }

    setLawToDelete(law)
  }

  async function confirmDeleteLaw() {
    if (!lawToDelete) return
    setIsDeleting(true)
    try {
      await lawState.deleteLaw(lawToDelete)
    } catch (e) {
      console.error(e)
    } finally {
      setIsDeleting(false)
      setLawToDelete(null)
    }
  }

  async function confirmDeleteSession() {
    if (!sessionToDelete) return
    setIsDeleting(true)
    try {
      await sessionState.deleteSession(Number(sessionToDelete))
      if (activeConversationId === sessionToDelete) {
        handleNewChat()
      }
    } catch (e) {
      console.error(e)
    } finally {
      setIsDeleting(false)
      setSessionToDelete(null)
    }
  }

  if (!session) return null

  const sidebar = (onCloseMobile?: () => void) => (
    <ChatSidebar
      session={session}
      conversations={conversations}
      activeConversationId={activeConversationId}
      viewMode={viewMode}
      canViewLaws={canViewLaws}
      onNewChat={handleNewChat}
      onSelectConversation={handleSelectConversation}
      onOpenLaws={handleOpenLaws}
      onOpenLawyerManagement={handleOpenLawyerManagement}
      isCreateLawyerOpen={isCreateLawyerOpen}
      onLogout={handleLogout}
      onCloseMobile={onCloseMobile}
      onPinConversation={async (id, isPinned) => {
        try {
          await sessionState.pinSession(Number(id), isPinned)
        } catch (e) {
          console.error(e)
        }
      }}
      onDeleteConversation={(id) => {
        setSessionToDelete(id)
      }}
      hasMoreSessions={sessionState.hasMore}
      isLoadingMoreSessions={sessionState.isLoadingMore}
      onLoadMoreSessions={sessionState.loadMoreSessions}
    />
  )

  return (
    <div
      className={cn(
        "grid h-dvh min-h-0 grid-cols-1 overflow-hidden lg:grid-cols-[300px_minmax(0,1fr)]",
        viewMode === "chat" && desktopSourcesOpen && "xl:grid-cols-[300px_minmax(0,1fr)_340px]"
      )}
    >
      <aside className="hidden min-h-0 overflow-hidden border-r border-sidebar-border lg:block">
        {sidebar()}
      </aside>

      <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
        <SheetContent side="left" className="h-dvh w-[300px] overflow-hidden p-0">
          {sidebar(() => setMobileNavOpen(false))}
        </SheetContent>
      </Sheet>

      {viewMode === "laws" && canViewLaws ? (
        <LawManager
          laws={lawState.laws}
          isLoadingLaws={lawState.isLoadingLaws}
          uploadMessage={lawState.uploadMessage}
          isUploading={lawState.isUploading}
          deletingLawId={lawState.deletingLawId}
          canManageLaws={canManageLaws}
          page={lawState.page}
          size={lawState.size}
          total={lawState.total}
          onPageChange={lawState.setPage}
          onIngest={handleIngest}
          onDelete={handleDelete}
          onOpenMobileMenu={() => setMobileNavOpen(true)}
          session={session}
          onLogout={handleLogout}
          onOpenSettings={(tab) => {
            setSettingsTab(tab)
            setIsSettingsOpen(true)
          }}
        />
      ) : (
        <ChatMain
          conversation={activeConversation}
          selectedTurnId={selectedTurnId}
          streamingTurnId={streamingTurnId}
          isAsking={chat.isAsking}
          pendingQuestion={
            pendingTurn?.conversationId === activeConversationId ? pendingTurn.question : ""
          }
          chatError={chat.chatError}
          query={query}
          serverState={health.serverState}
          session={session}
          onQueryChange={setQuery}
          onSubmit={handleAsk}
          onLogout={handleLogout}
          onSelectTurn={(id) => {
            setSelectedTurnId(id)
            if (window.innerWidth >= 1280) {
              setDesktopSourcesOpen(true)
            } else {
              setMobileSourcesOpen(true)
            }
          }}
          onStreamDone={() => setStreamingTurnId(null)}
          onOpenMobileMenu={() => setMobileNavOpen(true)}
          onOpenSources={() => setMobileSourcesOpen(true)}
          onFeedback={handleFeedback}
          onOpenSettings={(tab) => {
            setSettingsTab(tab)
            setIsSettingsOpen(true)
          }}
          hasMoreMessages={messagePages[activeConversationId]?.hasMore}
          isLoadingMoreMessages={isLoadingMoreMessages}
          onLoadMoreMessages={handleLoadMoreMessages}
        />
      )}

      {viewMode === "chat" && desktopSourcesOpen ? (
        <aside className="hidden min-h-0 overflow-hidden border-l border-border xl:block">
          <SourcesPanel selectedTurn={selectedTurn} onViewThinking={setThinkingSource} onClose={() => setDesktopSourcesOpen(false)} />
        </aside>
      ) : null}
      <Sheet open={mobileSourcesOpen} onOpenChange={setMobileSourcesOpen}>
        <SheetContent side="right" className="h-dvh w-[340px] max-w-[88vw] overflow-hidden p-0">
          <SourcesPanel selectedTurn={selectedTurn} onViewThinking={setThinkingSource} />
        </SheetContent>
      </Sheet>

      <ThinkingModal source={thinkingSource} onClose={() => setThinkingSource(null)} />
      {session.user.role === "super_admin" ? (
        <CreateLawyerDialog open={isCreateLawyerOpen} onOpenChange={setIsCreateLawyerOpen} />
      ) : null}

      <SettingsDialog 
        open={isSettingsOpen} 
        onOpenChange={setIsSettingsOpen} 
        session={session} 
        defaultTab={settingsTab} 
      />

      <ConfirmDialog
        open={!!lawToDelete}
        onOpenChange={(open) => { if (!open) setLawToDelete(null) }}
        title="আইন মুছে ফেলুন"
        description={`আপনি কি নিশ্চিত যে আপনি "${lawToDelete?.law_name || lawToDelete?.law_id}" মুছে ফেলতে চান?`}
        onConfirm={confirmDeleteLaw}
        isLoading={isDeleting}
      />
      <ConfirmDialog
        open={!!sessionToDelete}
        onOpenChange={(open) => { if (!open) setSessionToDelete(null) }}
        title="চ্যাট মুছে ফেলুন"
        description="আপনি কি নিশ্চিত যে আপনি এই চ্যাটটি মুছে ফেলতে চান? এটি আর ফিরিয়ে আনা যাবে না।"
        onConfirm={confirmDeleteSession}
        isLoading={isDeleting}
      />
    </div>
  )
}
