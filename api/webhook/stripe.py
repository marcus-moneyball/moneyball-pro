from http.server import BaseHTTPRequestHandler
import os
import json
import stripe

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        sig_header = self.headers.get('Stripe-Signature')

        try:
            event = stripe.Webhook.construct_event(
                post_data, sig_header, webhook_secret
            )
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            customer_email = session.get('customer_details', {}).get('email')
            # Lógica de liberação aqui

        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"received": True}).encode())
