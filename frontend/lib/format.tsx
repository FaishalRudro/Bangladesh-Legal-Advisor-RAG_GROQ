import type { ReactNode } from "react"

type InlineFormatRule = {
  regex: RegExp
  render: (match: RegExpMatchArray, key: string) => ReactNode
}

const inlineFormatRules: InlineFormatRule[] = [
  {
    regex: /`([^`]+)`/,
    render: (match, key) => <code key={key}>{match[1]}</code>,
  },
  {
    regex: /\*\*([\s\S]+?)\*\*/,
    render: (match, key) => <strong key={key}>{renderInlineText(match[1], key)}</strong>,
  },
  {
    regex: /__([\s\S]+?)__/,
    render: (match, key) => <strong key={key}>{renderInlineText(match[1], key)}</strong>,
  },
  {
    regex: /(?<!\*)\*([^*\n]+)\*(?!\*)/,
    render: (match, key) => <em key={key}>{renderInlineText(match[1], key)}</em>,
  },
  {
    regex: /(?<!_)_([^_\n]+)_(?!_)/,
    render: (match, key) => <em key={key}>{renderInlineText(match[1], key)}</em>,
  },
  {
    regex: /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/,
    render: (match, key) => (
      <a href={match[2]} key={key} rel="noreferrer" target="_blank">
        {renderInlineText(match[1], key)}
      </a>
    ),
  },
]

export function renderInlineText(text: string, keyPrefix = "inline"): ReactNode[] {
  if (!text) {
    return []
  }

  let earliest:
    | {
        rule: InlineFormatRule
        match: RegExpMatchArray
      }
    | null = null

  for (const rule of inlineFormatRules) {
    const flags = rule.regex.flags.replace("g", "")
    const regex = new RegExp(rule.regex.source, flags)
    const match = text.match(regex)
    const matchIndex = match?.index ?? -1

    if (
      match &&
      matchIndex >= 0 &&
      (!earliest || matchIndex < (earliest.match.index ?? Number.MAX_SAFE_INTEGER))
    ) {
      earliest = { rule, match }
    }
  }

  if (!earliest || earliest.match.index === undefined) {
    return [text]
  }

  const { rule, match } = earliest
  const matchIndex = match.index ?? 0
  const before = text.slice(0, matchIndex)
  const matched = match[0]
  const after = text.slice(matchIndex + matched.length)
  const key = `${keyPrefix}-${matchIndex}-${matched.length}`

  return [
    ...renderInlineText(before, `${key}-before`),
    rule.render(match, key),
    ...renderInlineText(after, `${key}-after`),
  ]
}

export function renderFormattedAnswer(text: string) {
  const lines = (text || "").split(/\r?\n/)

  return lines.map((line, index) => {
    const key = `line-${index}`

    if (!line.trim()) {
      return <div className="formatted-space" key={key} />
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/)
    if (heading) {
      const content = renderInlineText(heading[2], key)

      if (heading[1].length <= 1) {
        return (
          <h3 className="formatted-heading text-lg" key={key}>
            {content}
          </h3>
        )
      }

      if (heading[1].length <= 3) {
        return (
          <h4 className="formatted-heading text-base" key={key}>
            {content}
          </h4>
        )
      }

      return (
        <h5 className="formatted-heading text-sm" key={key}>
          {content}
        </h5>
      )
    }

    const bullet = line.match(/^\s*[-*•]\s+(.+)$/)
    if (bullet) {
      return (
        <p className="formatted-list-item" key={key}>
          <span>•</span>
          <span>{renderInlineText(bullet[1], key)}</span>
        </p>
      )
    }

    const ordered = line.match(/^\s*(\d+)[.)]\s+(.+)$/)
    if (ordered) {
      return (
        <p className="formatted-list-item" key={key}>
          <span>{ordered[1]}.</span>
          <span>{renderInlineText(ordered[2], key)}</span>
        </p>
      )
    }

    return <p key={key}>{renderInlineText(line, key)}</p>
  })
}

export function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function formatDate(value: string) {
  if (!value) {
    return ""
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return value
  }

  return new Intl.DateTimeFormat("bn-BD", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date)
}

export function percent(value: number) {
  return `${Math.round(value * 100)}%`
}
