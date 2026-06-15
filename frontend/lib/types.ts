export type HealthResponse = {
  status: string;
  message: string;
};

export type UserRole = "super_admin" | "Lawyer" | "user";

export type UserCreate = {
  name: string;
  email: string;
  password: string;
};

export type LawyerCreate = {
  name: string;
  email: string;
};

export type AcceptInvite = {
  email: string;
  password: string;
};

export type UserLogin = {
  email: string;
  password: string;
};

export type VerifyOTP = {
  email: string;
  otp: string;
};

export type ResendOTP = {
  email: string;
};

export type ForgotPassword = {
  email: string;
};

export type ResetPassword = {
  token: string;
  new_password: string;
};

export type UserResponse = {
  id: number;
  name: string;
  email: string;
  role: UserRole;
  is_active: boolean;
  is_verified: boolean;
};

export type Token = {
  access_token: string;
  token_type: string;
  user: UserResponse;
};

export type AuthSession = {
  accessToken: string;
  tokenType: string;
  user: UserResponse;
};

export type LawInfo = {
  law_id: string;
  law_name: string;
  year: string;
  link: string;
  repealed: string;
  total_chunks: number;
  ingested_at: string;
};

export type LawListResponse = {
  laws: LawInfo[];
  total: number;
  page: number;
  size: number;
};

export type LawIngestResponse = {
  status: "started" | "failed";
  message: string;
};

export type Source = {
  law_name: string;
  year: string;
  link: string;
  repealed: string;
  relevance_score: number;
  relevance_label: "high" | "medium" | "low";
  context_text?: string;
};

export type ChatRequest = {
  query: string;
  law_id: string | null;
  top_k: number;
  session_id?: number;
};

export type ChatResponse = {
  message_id?: number;
  session_id?: number;
  answer: string;
  sources: Source[];
  confidence: "high" | "medium" | "low" | "not_found";
  query_language: "bn" | "en";
  legal_query: string;
  laws_searched: string[];
  total_chunks_retrieved: number;
};

export type SessionSummary = {
  id: number;
  user_id: number;
  title: string;
  created_at: string;
  is_pinned: boolean;
};

export type SessionListResponse = {
  total: number;
  page: number;
  size: number;
  sessions: SessionSummary[];
};

export type Feedback = {
  id?: number;
  is_good: boolean | null;
  feedback_text: string | null;
  lawyer_name?: string;
};

export type ChatMessage = ChatResponse & {
  id?: number;
  message_id?: number;
  query?: string;
  user_query?: string;
  created_at: string;
  feedbacks?: Feedback[];
  ai_response?: any;
};

export type SessionDetailResponse = {
  id: number;
  user_id: number;
  title: string;
  created_at: string;
  is_pinned: boolean;
  total_messages: number;
  page: number;
  size: number;
  messages: ChatMessage[];
};
