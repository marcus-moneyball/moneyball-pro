import { NextResponse } from 'next/server';
import jwt from 'jsonwebtoken';

export async function POST(req: Request) {
  try {
    const { email } = await req.json();

    if (!email) {
      return NextResponse.json({ authorized: false, reason: 'E-mail não fornecido' }, { status: 400 });
    }

    const ghostUrl = process.env.GHOST_URL;
    const adminKey = process.env.GHOST_ADMIN_API_KEY;

    if (!ghostUrl || !adminKey) {
      return NextResponse.json({ error: 'Configuração do Ghost ausente' }, { status: 500 });
    }

    // Gerar token JWT temporário para autenticar na Admin API do Ghost
    const [id, secret] = adminKey.split(':');
    const token = jwt.sign({}, Buffer.from(secret, 'hex'), {
      keyid: id,
      algorithm: 'HS256',
      expiresIn: '5m',
      audience: `/v5/admin/`,
    });

    // Consultar o membro no Ghost pelo e-mail
    const res = await fetch(`${ghostUrl}/ghost/api/v5/admin/members/?filter=email:'${email}'`, {
      headers: {
        Authorization: `Ghost ${token}`,
      },
    });

    const data = await res.json();
    const member = data.members?.[0];

    // Verifica se o membro existe e se o status da assinatura é "paid" (pago)
    if (member && member.status === 'paid') {
      return NextResponse.json({ authorized: true, status: member.status, email: member.email });
    }

    return NextResponse.json({ authorized: false, reason: 'Assinatura inativa ou inexistente' });
  } catch (error: any) {
    return NextResponse.json({ authorized: false, error: error.message }, { status: 500 });
  }
}
