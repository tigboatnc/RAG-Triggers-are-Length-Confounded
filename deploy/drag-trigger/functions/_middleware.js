const REALM = 'Basic realm="Recollection — Dynamic RAG Triggers"';

export async function onRequest(context) {
  const { request, next, env } = context;

  const expectedUser = env.BASIC_AUTH_USER;
  const expectedPass = env.BASIC_AUTH_PASS;

  if (!expectedUser || !expectedPass) {
    return new Response(
      "Auth not configured. Run ./deploy/deploy.sh (it reads deploy/drag-trigger/.dev.vars).",
      { status: 503, headers: { "Content-Type": "text/plain; charset=utf-8" } }
    );
  }

  const authHeader = request.headers.get("Authorization");
  if (!authHeader) {
    return new Response("Unauthorized", {
      status: 401,
      headers: { "WWW-Authenticate": REALM },
    });
  }

  const [scheme, encoded] = authHeader.split(" ");
  if (scheme !== "Basic" || !encoded) {
    return new Response("Malformed authorization", { status: 400 });
  }

  try {
    const decoded = atob(encoded);
    const colon = decoded.indexOf(":");
    const user = colon >= 0 ? decoded.slice(0, colon) : decoded;
    const pass = colon >= 0 ? decoded.slice(colon + 1) : "";

    if (user === expectedUser && pass === expectedPass) {
      return next();
    }
  } catch {
    return new Response("Invalid request", { status: 400 });
  }

  return new Response("Invalid credentials", {
    status: 401,
    headers: { "WWW-Authenticate": REALM },
  });
}
