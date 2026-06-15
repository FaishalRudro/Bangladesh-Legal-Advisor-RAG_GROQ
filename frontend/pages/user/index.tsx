import Head from "next/head"
import dynamic from "next/dynamic"
import { withAuth } from "@/lib/with-auth"

import { PageLoader } from "@/components/page-loader"

const ChatApp = dynamic(() => import("@/components/chat-app").then((m) => m.ChatApp), {
  ssr: false,
  loading: () => <PageLoader />,
})

function UserPage() {
  return (
    <>
      <Head>
        <title>Bangladesh Legal Advisor</title>
      </Head>
      <ChatApp />
    </>
  )
}

export default withAuth(UserPage, ["user"])
