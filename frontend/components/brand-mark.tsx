import { cn } from "@/lib/utils"
import { Scale } from "lucide-react"

export function BrandMark({
  size = 40,
  className,
}: {
  size?: number
  className?: string
}) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center justify-center overflow-hidden rounded-xl bg-white ring-1 ring-border",
        className,
      )}
      style={{ width: size, height: size }}
    >
      <Scale size={size * 0.6} className="text-[#0284c7]" />
    </span>
  )
}
