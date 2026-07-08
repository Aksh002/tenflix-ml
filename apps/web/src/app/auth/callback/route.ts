import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const next = url.searchParams.get("next") ?? "/app";
  const errorDescription =
    url.searchParams.get("error_description") ?? url.searchParams.get("error");
  if (errorDescription) {
    const loginUrl = new URL("/login", url.origin);
    loginUrl.searchParams.set("auth_error", errorDescription);
    return NextResponse.redirect(loginUrl);
  }
  if (code) {
    const supabase = await createClient();
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (error) {
      const loginUrl = new URL("/login", url.origin);
      loginUrl.searchParams.set("auth_error", error.message);
      return NextResponse.redirect(loginUrl);
    }
    return NextResponse.redirect(new URL(next, url.origin));
  }
  const loginUrl = new URL("/login", url.origin);
  loginUrl.searchParams.set("auth_error", "Missing auth code in Supabase callback");
  return NextResponse.redirect(loginUrl);
}
