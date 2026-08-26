import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // Exemplo usando verificação de cookie de sessão/token
  const authCookie = request.cookies.get('mbp_access_token');
  const { pathname } = request.nextUrl;

  // Protege a rota do produto /dashboard ou /app
  if (pathname.startsWith('/app') || pathname.startsWith('/dashboard')) {
    if (!authCookie) {
      // Redireciona para a VSL / Checkout caso não esteja autenticado
      return NextResponse.redirect(new URL('/#checkout', request.url));
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/app/:path*', '/dashboard/:path*'],
};
