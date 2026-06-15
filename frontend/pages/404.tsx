import Head from "next/head"
import { useRouter } from "next/router"
import { BrandMark } from "@/components/brand-mark"
import { Button } from "@/components/ui/button"
import { ShieldAlert } from "lucide-react"

export default function NotFoundPage() {
  const router = useRouter()

  return (
    <>
      <Head>
        <title>Page Not Found — Bangladesh Legal Advisor</title>
      </Head>
      <main className="relative min-h-dvh overflow-hidden bg-gradient-to-b from-[#e0f2fe] via-[#bae6fd] to-white flex items-center justify-center px-6 py-8">
        <div className="w-full max-w-xl animate-in fade-in zoom-in-95 duration-500">
          <div className="mb-8 flex flex-col items-center text-center">
            <BrandMark size={64} />
            <p className="mt-3 text-sm text-gray-700">Bangladesh Legal Advisor</p>
            <h1 className="font-heading text-3xl font-bold text-gray-900">Bangladesh Legal Advisor</h1>
          </div>

          <div className="rounded-2xl border-2 border-white/60 bg-white/30 p-10 sm:p-14 shadow-xl backdrop-blur-2xl text-center">
            <span className="inline-flex size-16 items-center justify-center rounded-full bg-[#FFEAE5] text-destructive mb-4">
              <ShieldAlert className="size-8" />
            </span>
            <h2 className="font-heading text-2xl font-bold text-gray-800">404 - Not Found</h2>
            <p className="mt-2 text-sm text-gray-700">
              The page you are looking for does not exist, or you do not have permission to access it.
            </p>
            <Button
              type="button"
              className="mt-6 h-11 w-full rounded-xl text-base bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer"
              onClick={() => router.replace("/")}
            >
              Go to Home
            </Button>
          </div>
        </div>

        {/* Copyright footer - bottom center */}
        <div className="absolute bottom-0 left-0 right-0 pb-4 text-center">
          <p className="text-xs font-bold text-gray-500">
            © {new Date().getFullYear()} Bangladesh Legal Advisor. All rights reserved.
          </p>
        </div>
      </main>
    </>
  )
}
