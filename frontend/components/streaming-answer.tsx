import { useEffect, useState } from "react"
import { renderFormattedAnswer } from "@/lib/format"

export function StreamingAnswer({
  shouldStream,
  text,
  onDone,
}: {
  shouldStream: boolean
  text: string
  onDone: () => void
}) {
  const [visibleText, setVisibleText] = useState(shouldStream ? "" : text)

  useEffect(() => {
    if (!shouldStream) {
      setVisibleText(text)
      return
    }

    let index = 0
    let timer: ReturnType<typeof setTimeout> | null = null
    const stepSize = Math.max(2, Math.ceil(text.length / 180))

    function tick() {
      index = Math.min(text.length, index + stepSize)
      setVisibleText(text.slice(0, index))

      if (index < text.length) {
        timer = setTimeout(tick, 18)
      } else {
        onDone()
      }
    }

    setVisibleText("")
    timer = setTimeout(tick, 80)

    return () => {
      if (timer) {
        clearTimeout(timer)
      }
    }
  }, [onDone, shouldStream, text])

  return (
    <div className="answer-prose text-[0.95rem] text-foreground text-justify" dir="auto">
      {renderFormattedAnswer(visibleText)}
      {shouldStream ? <span className="stream-caret" aria-hidden="true" /> : null}
    </div>
  )
}
