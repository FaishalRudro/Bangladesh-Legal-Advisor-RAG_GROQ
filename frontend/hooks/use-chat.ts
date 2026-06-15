import { useCallback, useState } from "react"
import { chatApi } from "@/lib/api/chat"
import { getApiErrorMessage } from "@/lib/http"
import type { ChatRequest } from "@/lib/types"

export function useChat() {
  const [isAsking, setIsAsking] = useState(false)
  const [chatError, setChatError] = useState("")

  const ask = useCallback(async (payload: ChatRequest) => {
    setChatError("")
    setIsAsking(true)

    try {
      return await chatApi.ask(payload)
    } catch (error) {
      const message = getApiErrorMessage(error)
      setChatError(message)
      throw error
    } finally {
      setIsAsking(false)
    }
  }, [])

  return {
    isAsking,
    chatError,
    setChatError,
    ask,
  }
}
