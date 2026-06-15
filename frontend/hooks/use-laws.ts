import { useCallback, useEffect, useState } from "react"
import { lawsApi } from "@/lib/api/laws"
import { getApiErrorMessage } from "@/lib/http"
import type { LawInfo } from "@/lib/types"

export function useLaws(enabled: boolean) {
  const [laws, setLaws] = useState<LawInfo[]>([])
  const [isLoadingLaws, setIsLoadingLaws] = useState(false)
  const [uploadMessage, setUploadMessage] = useState("")
  const [isUploading, setIsUploading] = useState(false)
  const [deletingLawId, setDeletingLawId] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [size, setSize] = useState(10)
  const [total, setTotal] = useState(0)

  const loadLaws = useCallback(async () => {
    if (!enabled) return

    setIsLoadingLaws(true)

    try {
      const response = await lawsApi.list(page, size)
      
      if (response.laws.length === 0 && response.total > 0 && page > 1) {
        setPage(Math.max(1, Math.ceil(response.total / size)))
        return
      }
      
      setLaws(response.laws)
      setTotal(response.total)
    } catch (error) {
      setUploadMessage(getApiErrorMessage(error))
    } finally {
      setIsLoadingLaws(false)
    }
  }, [enabled, page, size])

  useEffect(() => {
    if (!enabled) {
      setLaws([])
      setUploadMessage("")
      return
    }

    loadLaws()
  }, [enabled, loadLaws])

  const ingestLaws = useCallback(
    async (limit: number) => {
      setIsUploading(true)
      setUploadMessage("")

      try {
        const response = await lawsApi.ingest(limit)
        setUploadMessage(response.message)
        await loadLaws()
      } catch (error) {
        setUploadMessage(getApiErrorMessage(error))
      } finally {
        setIsUploading(false)
      }
    },
    [loadLaws],
  )

  const deleteLaw = useCallback(
    async (law: LawInfo) => {
      setDeletingLawId(law.law_id)
      setUploadMessage("")

      try {
        await lawsApi.delete(law.law_id)
        await loadLaws()
      } catch (error) {
        setUploadMessage(getApiErrorMessage(error))
      } finally {
        setDeletingLawId(null)
      }
    },
    [loadLaws],
  )

  return {
    laws,
    isLoadingLaws,
    uploadMessage,
    isUploading,
    deletingLawId,
    page,
    size,
    total,
    setPage,
    setSize,
    setUploadMessage,
    loadLaws,
    ingestLaws,
    deleteLaw,
  }
}
