import { BrandMark } from "@/components/brand-mark"
import { Button } from "@/components/ui/button"
import { BookOpen, Menu, MessageCircleQuestion, ShieldCheck, Sparkles, Star, Users, X, Zap } from "lucide-react"
import Link from "next/link"
import { useState } from "react"
import { useLanguage } from "@/lib/language-context"

type Language = "bn" | "en"

const translations = {
  bn: {
    tagline: "এআই চালিত আইনি জ্ঞান সহায়ক",
    heroTitle: "বিশ্বাসযোগ্য আইনি জ্ঞান",
    heroHighlight: " আপনার হাতের মুঠোয়",
    heroDesc: "আইন, ধারা এবং অধ্যাদেশ সমূহের রেফারেন্স সহ আপনার প্রশ্নের সঠিক উত্তর পান। যাচাইকৃত তথ্যসূত্র থেকে নির্ভরযোগ্য জ্ঞান অর্জন করুন।",
    getStarted: "শুরু করুন",
    features: "",
    howItWorks: "",
    pricing: "মূল্য",
    faq: "প্রশ্ন",
    whyIslamicGPT: "কেন Bangladesh Legal Advisor?",
    whyDesc: "আমাদের প্ল্যাটফর্ম আইনি জ্ঞান অর্জনকে সহজ এবং নির্ভরযোগ্য করে তোলে",
    feature1Title: "সহজ প্রশ্ন করুন",
    feature1Desc: "বাংলা বা ইংরেজিতে যেকোনো আইনি প্রশ্ন করুন এবং তাৎক্ষণিক উত্তর পান",
    feature2Title: "বিশ্বস্ত রেফারেন্স",
    feature2Desc: "প্রতিটি উত্তরের সাথে আইন, ধারা এবং অধ্যাদেশ সমূহের রেফারেন্স দেওয়া হয়",
    feature3Title: "যাচাইকৃত তথ্য",
    feature3Desc: "লইয়ারদের দ্বারা যাচাইকৃত তথ্য এবং নির্ভরযোগ্য আইনি সোর্স থেকে উত্তর",
    howTitle: "",
    howDesc: "মাত্র তিনটি সহজ ধাপে আপনার প্রশ্নের উত্তর পান",
    step1: "অ্যাকাউন্ট তৈরি করুন",
    step1Desc: "বিনামূল্যে সাইন আপ করুন এবং ইমেইল যাচাই করুন",
    step2: "প্রশ্ন জিজ্ঞাসা করুন",
    step2Desc: "আপনার আইনি প্রশ্ন বাংলা বা ইংরেজিতে লিখুন",
    step3: "উত্তর পান",
    step3Desc: "রেফারেন্স সহ বিস্তারিত এবং নির্ভরযোগ্য উত্তর পান",
  },
  en: {
    tagline: "AI-Powered Legal Knowledge Assistant",
    heroTitle: "Trusted Legal Knowledge",
    heroHighlight: " at Your Fingertips",
    heroDesc: "Get accurate answers to your questions with references from Laws, Acts, and Ordinances. Obtain reliable knowledge from verified sources.",
    getStarted: "Get Started",
    features: "",
    howItWorks: "How It Works",
    pricing: "Pricing",
    faq: "FAQ",
    whyIslamicGPT: "Why Bangladesh Legal Advisor?",
    whyDesc: "Our platform makes acquiring Legal knowledge simple and reliable",
    feature1Title: "Ask Questions Easily",
    feature1Desc: "Ask any Legal question in Bengali or English and get instant answers",
    feature2Title: "Trusted References",
    feature2Desc: "Every answer includes references from Laws, Acts, and Ordinances",
    feature3Title: "Verified Information",
    feature3Desc: "Answers from Lawyer-verified information and reliable Legal sources",
    howTitle: "How It Works",
    howDesc: "Get answers to your questions in just three simple steps",
    step1: "Create Account",
    step1Desc: "Sign up for free and verify your email",
    step2: "Ask Questions",
    step2Desc: "Write your Legal question in Bengali or English",
    step3: "Get Answers",
    step3Desc: "Receive detailed and reliable answers with references",
  },
}

export function LandingPage() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const { language, setLanguage } = useLanguage()
  
  const t = translations[language]

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#e0f2fe] to-white">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-gray-200 bg-white/80 backdrop-blur-lg">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 md:px-6 md:py-4">
          <div className="flex items-center gap-2 md:gap-3">
            <BrandMark size={32} className="md:hidden" />
            <BrandMark size={40} className="hidden md:block" />
            <div className="leading-tight">
              <p className="text-[10px] text-gray-600 md:text-xs">Bangladesh Legal Advisor</p>
              <p className="font-heading text-sm font-bold text-gray-900 md:text-lg">Bangladesh Legal Advisor</p>
            </div>
          </div>
          
          {/* Desktop Navigation */}
          <nav className="hidden items-center gap-8 md:flex">
            {/* Language Toggle */}
            <div className="flex h-11 items-center gap-1 rounded border-2 border-gray-300 p-1">
              <button
                onClick={() => setLanguage("bn")}
                className={`rounded px-3 py-2 text-sm font-medium transition cursor-pointer ${
                  language === "bn" ? "bg-[#0284c7] text-white" : "text-gray-700 hover:text-gray-900"
                }`}
              >
                বাংলা
              </button>
              <button
                onClick={() => setLanguage("en")}
                className={`rounded px-3 py-2 text-sm font-medium transition cursor-pointer ${
                  language === "en" ? "bg-[#0284c7] text-white" : "text-gray-700 hover:text-gray-900"
                }`}
              >
                English
              </button>
            </div>

            <Link href="/login">
              <Button size="lg" className="h-11 bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer shadow-lg hover:shadow-xl transition-all hover:scale-105 rounded px-6 text-base font-semibold">{t.getStarted}</Button>
            </Link>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 text-gray-700 hover:text-[#0284c7] transition cursor-pointer"
          >
            {mobileMenuOpen ? <X className="size-6" /> : <Menu className="size-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="border-t border-b-2 border-gray-200 bg-white shadow-lg md:hidden">
            <nav className="flex flex-col px-4 py-4 divide-y divide-gray-200">
              
              {/* Language Toggle Mobile */}
              <div className="py-3">
                <div className="flex items-center gap-1 rounded-lg border-2 border-gray-300 p-1">
                  <button
                    onClick={() => setLanguage("bn")}
                    className={`flex-1 rounded px-3 py-2 text-sm font-medium transition cursor-pointer ${
                      language === "bn" ? "bg-[#0284c7] text-white" : "text-gray-700"
                    }`}
                  >
                    বাংলা
                  </button>
                  <button
                    onClick={() => setLanguage("en")}
                    className={`flex-1 rounded px-3 py-2 text-sm font-medium transition cursor-pointer ${
                      language === "en" ? "bg-[#0284c7] text-white" : "text-gray-700"
                    }`}
                  >
                    English
                  </button>
                </div>
              </div>

              <div className="pt-3">
                <Link href="/login" onClick={() => setMobileMenuOpen(false)}>
                  <Button size="lg" className="w-full bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer rounded-md text-base font-semibold">{t.getStarted}</Button>
                </Link>
              </div>
            </nav>
          </div>
        )}
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-b from-[#e0f2fe] via-[#bae6fd] to-white px-4 py-12 md:px-6 md:py-24"> 
        {/* Decorative elements */}
        <div className="pointer-events-none absolute inset-0 opacity-20">
          <svg width="100%" height="100%" className="absolute">
            <defs>
              <pattern id="Legal-pattern" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
                <circle cx="50" cy="50" r="2" fill="#0284c7" opacity="0.3" />
                <path d="M50,30 Q60,40 50,50 Q40,40 50,30 M50,70 Q60,60 50,50 Q40,60 50,70" stroke="#0284c7" fill="none" opacity="0.2" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#Legal-pattern)" />
          </svg>
        </div>

        <div className="relative mx-auto max-w-6xl text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border-2 border-[#0284c7]/30 bg-[#0284c7]/10 px-3 py-1.5 md:mb-6 md:px-4 md:py-2">
            <Sparkles className="size-4 text-[#0284c7] md:size-5" />
            <span className="text-xs font-semibold text-gray-800 md:text-sm">{t.tagline}</span>
          </div>
          
          <h1 className="mx-auto mb-4 max-w-4xl font-heading text-3xl font-bold leading-tight text-gray-900 md:mb-6 md:text-5xl lg:text-6xl">
            {t.heroTitle}
            <span className="text-[#0284c7]">{t.heroHighlight}</span>
          </h1>
          
          <p className="mx-auto mb-8 max-w-2xl text-base leading-relaxed text-gray-700 md:mb-16 md:text-lg">
            {t.heroDesc}
          </p>

          {/* <div className="mt-16 flex items-center justify-center gap-8 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <Users className="size-5 text-[#0284c7]" />
              <span>১০,০০০+ ব্যবহারকারী</span>
            </div>
            <div className="flex items-center gap-2">
              <Star className="size-5 text-[#0284c7]" />
              <span>৫০,০০০+ প্রশ্নের উত্তর</span>
            </div>
          </div> */}
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="bg-white px-4 py-12 md:px-6 md:py-24">
        <div className="mx-auto max-w-6xl">
          <div className="mb-8 text-center md:mb-16">
            <h2 className="mb-3 font-heading text-3xl font-bold text-gray-900 md:mb-4 md:text-4xl">{t.whyIslamicGPT}</h2>
            <p className="mx-auto max-w-2xl text-base text-gray-700 md:text-lg">
              {t.whyDesc}
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-3 md:gap-8">
            <div className="rounded-2xl border-2 border-gray-200 bg-gradient-to-br from-[#e0f2fe] to-white p-8 shadow-lg transition hover:shadow-xl">
              <div className="mb-4 inline-flex size-14 items-center justify-center rounded-xl bg-[#0284c7]/20">
                <MessageCircleQuestion className="size-7 text-[#0284c7]" />
              </div>
              <h3 className="mb-3 font-heading text-xl font-bold text-gray-900">{t.feature1Title}</h3>
              <p className="text-gray-700">
                {t.feature1Desc}
              </p>
            </div>

            <div className="rounded-2xl border-2 border-gray-200 bg-gradient-to-br from-[#e0f2fe] to-white p-8 shadow-lg transition hover:shadow-xl">
              <div className="mb-4 inline-flex size-14 items-center justify-center rounded-xl bg-[#0284c7]/20">
                <BookOpen className="size-7 text-[#0284c7]" />
              </div>
              <h3 className="mb-3 font-heading text-xl font-bold text-gray-900">{t.feature2Title}</h3>
              <p className="text-gray-700">
                {t.feature2Desc}
              </p>
            </div>

            <div className="rounded-2xl border-2 border-gray-200 bg-gradient-to-br from-[#e0f2fe] to-white p-8 shadow-lg transition hover:shadow-xl">
              <div className="mb-4 inline-flex size-14 items-center justify-center rounded-xl bg-[#0284c7]/20">
                <ShieldCheck className="size-7 text-[#0284c7]" />
              </div>
              <h3 className="mb-3 font-heading text-xl font-bold text-gray-900">{t.feature3Title}</h3>
              <p className="text-gray-700">
                {t.feature3Desc}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="bg-gradient-to-b from-white to-[#e0f2fe] px-4 py-12 md:px-6 md:py-24">
        <div className="mx-auto max-w-6xl">
          <div className="mb-8 text-center md:mb-16">
            <h2 className="mb-3 font-heading text-3xl font-bold text-gray-900 md:mb-4 md:text-4xl">{t.howTitle}</h2>
            <p className="mx-auto max-w-2xl text-base text-gray-700 md:text-lg">
              {t.howDesc}
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3 md:gap-12">
            <div className="text-center">
              <div className="mx-auto mb-6 flex size-16 items-center justify-center rounded-full bg-[#0284c7] text-2xl font-bold text-white">
                {language === "bn" ? "১" : "1"}
              </div>
              <h3 className="mb-3 font-heading text-xl font-bold text-gray-900">{t.step1}</h3>
              <p className="text-gray-700">
                {t.step1Desc}
              </p>
            </div>

            <div className="text-center">
              <div className="mx-auto mb-6 flex size-16 items-center justify-center rounded-full bg-[#0284c7] text-2xl font-bold text-white">
                {language === "bn" ? "২" : "2"}
              </div>
              <h3 className="mb-3 font-heading text-xl font-bold text-gray-900">{t.step2}</h3>
              <p className="text-gray-700">
                {t.step2Desc}
              </p>
            </div>

            <div className="text-center">
              <div className="mx-auto mb-6 flex size-16 items-center justify-center rounded-full bg-[#0284c7] text-2xl font-bold text-white">
                {language === "bn" ? "৩" : "3"}
              </div>
              <h3 className="mb-3 font-heading text-xl font-bold text-gray-900">{t.step3}</h3>
              <p className="text-gray-700">
                {t.step3Desc}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      {/* <section id="pricing" className="bg-white px-6 py-24">
        <div className="mx-auto max-w-6xl">
          <div className="mb-16 text-center">
            <h2 className="mb-4 font-heading text-4xl font-bold text-gray-900">মূল্য পরিকল্পনা</h2>
            <p className="mx-auto max-w-2xl text-lg text-gray-700">
              আপনার প্রয়োজন অনুযায়ী সঠিক প্ল্যান নির্বাচন করুন
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3">
            <div className="rounded-2xl border-2 border-gray-200 bg-white p-8 shadow-lg">
              <h3 className="mb-2 font-heading text-2xl font-bold text-gray-900">বিনামূল্যে</h3>
              <div className="mb-6">
                <span className="text-4xl font-bold text-gray-900">৳০</span>
                <span className="text-gray-600">/মাস</span>
              </div>
              <ul className="mb-8 space-y-3">
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">প্রতিদিন ১০টি প্রশ্ন</span>
                </li>
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">বেসিক রেফারেন্স</span>
                </li>
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">চ্যাট হিস্টরি</span>
                </li>
              </ul>
              <Link href="/login">
                <Button variant="outline" className="w-full cursor-pointer">শুরু করুন</Button>
              </Link>
            </div>

            <div className="relative rounded-2xl border-2 border-[#0284c7] bg-gradient-to-br from-[#e0f2fe] to-white p-8 shadow-xl">
              <div className="absolute -top-4 left-1/2 -translate-x-1/2 rounded-full bg-[#0284c7] px-4 py-1 text-sm font-bold text-white">
                জনপ্রিয়
              </div>
              <h3 className="mb-2 font-heading text-2xl font-bold text-gray-900">প্রো</h3>
              <div className="mb-6">
                <span className="text-4xl font-bold text-gray-900">৳৪৯৯</span>
                <span className="text-gray-600">/মাস</span>
              </div>
              <ul className="mb-8 space-y-3">
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">আনলিমিটেড প্রশ্ন</span>
                </li>
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">বিস্তারিত রেফারেন্স</span>
                </li>
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">দ্রুত উত্তর</span>
                </li>
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">প্রাইঅরিটি সাপোর্ট</span>
                </li>
              </ul>
              <Link href="/login">
                <Button className="w-full bg-[#0284c7] hover:bg-[#0284c7]/90 cursor-pointer">শুরু করুন</Button>
              </Link>
            </div>

            <div className="rounded-2xl border-2 border-gray-200 bg-white p-8 shadow-lg">
              <h3 className="mb-2 font-heading text-2xl font-bold text-gray-900">এন্টারপ্রাইজ</h3>
              <div className="mb-6">
                <span className="text-4xl font-bold text-gray-900">কাস্টম</span>
              </div>
              <ul className="mb-8 space-y-3">
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">সকল প্রো ফিচার</span>
                </li>
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">কাস্টম ইন্টিগ্রেশন</span>
                </li>
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">ডেডিকেটেড সাপোর্ট</span>
                </li>
                <li className="flex items-start gap-2">
                  <Zap className="mt-0.5 size-5 shrink-0 text-[#0284c7]" />
                  <span className="text-gray-700">টিম ম্যানেজমেন্ট</span>
                </li>
              </ul>
              <Button variant="outline" className="w-full cursor-pointer">যোগাযোগ করুন</Button>
            </div>
          </div>
        </div>
      </section> */}

      {/* FAQ Section */}
      {/* <section id="faq" className="bg-gradient-to-b from-white to-[#e0f2fe] px-6 py-24">
        <div className="mx-auto max-w-4xl">
          <div className="mb-16 text-center">
            <h2 className="mb-4 font-heading text-4xl font-bold text-gray-900">সাধারণ প্রশ্ন</h2>
            <p className="text-lg text-gray-700">আপনার প্রশ্নের উত্তর এখানে খুঁজে নিন</p>
          </div>

          <div className="space-y-6">
            <div className="rounded-xl border-2 border-gray-200 bg-white p-6 shadow-sm">
              <h3 className="mb-2 font-heading text-lg font-bold text-gray-900">Bangladesh Legal Advisor কি?</h3>
              <p className="text-gray-700">
                Bangladesh Legal Advisor হলো একটি এআই চালিত প্ল্যাটফর্ম যা কুরআন, হাদিস এবং ইসলামিক স্কলারদের রেফারেন্স ব্যবহার করে ইসলামিক প্রশ্নের নির্ভরযোগ্য উত্তর প্রদান করে।
              </p>
            </div>

            <div className="rounded-xl border-2 border-gray-200 bg-white p-6 shadow-sm">
              <h3 className="mb-2 font-heading text-lg font-bold text-gray-900">তথ্য কতটা নির্ভরযোগ্য?</h3>
              <p className="text-gray-700">
                আমাদের সকল উত্তর মুফতি এবং ইসলামিক স্কলারদের দ্বারা যাচাইকৃত তথ্যসূত্র থেকে প্রদান করা হয়। প্রতিটি উত্তরের সাথে রেফারেন্স দেওয়া থাকে।
              </p>
            </div>

            <div className="rounded-xl border-2 border-gray-200 bg-white p-6 shadow-sm">
              <h3 className="mb-2 font-heading text-lg font-bold text-gray-900">কোন ভাষায় প্রশ্ন করতে পারি?</h3>
              <p className="text-gray-700">
                আপনি বাংলা এবং ইংরেজি উভয় ভাষায় প্রশ্ন করতে পারবেন। আমাদের সিস্টেম উভয় ভাষায় সমান ভালো কাজ করে।
              </p>
            </div>

            <div className="rounded-xl border-2 border-gray-200 bg-white p-6 shadow-sm">
              <h3 className="mb-2 font-heading text-lg font-bold text-gray-900">বিনামূল্যে ট্রায়াল আছে কি?</h3>
              <p className="text-gray-700">
                হ্যাঁ! আপনি বিনামূল্যে প্ল্যানে সাইন আপ করে প্রতিদিন ১০টি প্রশ্ন করতে পারবেন। কোনো ক্রেডিট কার্ডের প্রয়োজন নেই।
              </p>
            </div>
          </div>
        </div>
      </section> */}

      {/* Footer */}
      <footer className="bg-gray-900 px-6 py-12 text-gray-300">
        <div className="mx-auto max-w-6xl">
          <div className="grid gap-8 md:grid-cols-3">
            <div>
              <div className="mb-4 flex items-center gap-3">
                <BrandMark size={36} />
                <div className="leading-tight">
                  <p className="text-xs text-gray-400">Bangladesh Legal Advisor</p>
                  <p className="font-heading text-base font-bold text-white">Bangladesh Legal Advisor</p>
                </div>
              </div>
              <p className="text-sm text-gray-400">
                বিশ্বাসযোগ্য আইনি জ্ঞান যাচাইকৃত প্রবেশাধিকারে
              </p>
            </div>

            <div>
              <h4 className="mb-4 font-heading font-bold text-white">কোম্পানি</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="#" className="transition hover:text-[#0284c7]">আমাদের সম্পর্কে</a></li>
                <li><a href="#" className="transition hover:text-[#0284c7]">যোগাযোগ</a></li>
                <li><a href="#" className="transition hover:text-[#0284c7]">ব্লগ</a></li>
              </ul>
            </div>

            <div>
              <h4 className="mb-4 font-heading font-bold text-white">আইনি</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="#" className="transition hover:text-[#0284c7]">প্রাইভেসি পলিসি</a></li>
                <li><a href="#" className="transition hover:text-[#0284c7]">টার্মস অফ সার্ভিস</a></li>
              </ul>
            </div>
          </div>

          <div className="mt-12 border-t border-gray-800 pt-8 text-center text-sm">
            <p>© {new Date().getFullYear()} Bangladesh Legal Advisor. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
