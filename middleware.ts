import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Aplica o bloqueio apenas nas rotas do app / dashboard
  if (pathname.startsWith('/app') || pathname.startsWith('/dashboard')) {
    
    // Procura o cookie de sessão do assinante (ex: mbp_user_email ou token de sessão)
    const userEmail = request.cookies.get('mbp_user_email')?.value;

    if (!userEmail) {
      // Se não tiver o cookie de login, manda de volta para a landing page / login do Ghost
      return NextResponse.redirect(new URL('https://moneyballpro.com.br/#/#portal/signin', request.url));
    }

    // Opcional: faz a requisição de verificação no endpoint interno
    const checkUrl = new URL('/api/auth/check-subscription', request.url);
    const authRes = await fetch(checkUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: userEmail }),
    });

    const authData = await authRes.json();

    if (!authData.authorized) {
      // Se não for pagante ativo no Ghost, redireciona para a página de vendas
      return NextResponse.redirect(new URL('https://moneyballpro.com.br/#/#portal/signup', request.url));
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/app/:path*', '/dashboard/:path*'],
};
