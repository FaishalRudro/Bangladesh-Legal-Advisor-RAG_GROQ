import type { AppProps } from "next/app"
import { Inter, Plus_Jakarta_Sans, Hind_Siliguri, Geist_Mono } from "next/font/google"
import { AuthProvider } from "@/lib/auth-context"
import { LanguageProvider } from "@/lib/language-context"
import "@/app/globals.css"

const inter = Inter({ variable: "--font-inter", subsets: ["latin"], display: "swap" })
const jakarta = Plus_Jakarta_Sans({ variable: "--font-jakarta", subsets: ["latin"], display: "swap" })
const bengali = Hind_Siliguri({
  variable: "--font-bengali",
  subsets: ["bengali", "latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
})
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] })

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div
      suppressHydrationWarning
      className={`${inter.variable} ${jakarta.variable} ${bengali.variable} ${geistMono.variable} bg-background font-sans`}
    >
      <LanguageProvider>
        <AuthProvider>
          <Component {...pageProps} />
        </AuthProvider>
      </LanguageProvider>
    </div>
  )
}
