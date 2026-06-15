import axios, { type AxiosRequestConfig } from "axios";

export const api = axios.create({
  baseURL: "/api/backend",
  timeout: 300000
});

let authToken: string | null = null;

export function setAuthToken(token: string | null) {
  authToken = token;
}

api.interceptors.request.use((config) => {
  if (authToken) {
    config.headers.Authorization = `Bearer ${authToken}`;
  } else {
    delete config.headers.Authorization;
  }

  return config;
});

export async function apiRequest<T>(config: AxiosRequestConfig) {
  const response = await api.request<T>(config);

  return response.data;
}

export function getApiErrorMessage(error: unknown) {
  if (axios.isAxiosError(error)) {
    const payload = error.response?.data;
    
    if (typeof payload === "string") {
      return payload;
    }

    const detail = payload?.detail;
    const message = payload?.message;

    if (typeof detail === "string") {
      return detail;
    }

    if (Array.isArray(detail)) {
      return detail
        .map((item) => item?.msg || JSON.stringify(item))
        .filter(Boolean)
        .join(" ");
    }

    if (typeof message === "string") {
      return message;
    }

    if (typeof detail === "object" && detail !== null) {
      return JSON.stringify(detail);
    }

    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "Unexpected error.";
}
