const SEARCH_AND_AI_BOTS =
  /googlebot|google-inspectiontool|bingbot|adidxbot|yandexbot|baiduspider|duckduckbot|naverbot|yeti|daumoa|applebot|gptbot|chatgpt-user|oai-searchbot|claudebot|claude-web|anthropic-ai|perplexitybot|ccbot|amazonbot|bytespider|petalbot|semrushbot|ahrefsbot|mj12bot|dotbot/i;
export async function onRequest({ request, next }) {
  const userAgent = request.headers.get("user-agent") || "";
  if (SEARCH_AND_AI_BOTS.test(userAgent)) {
    return new Response("Bot access denied.", { status: 403, headers: { "Content-Type": "text/plain; charset=utf-8", "X-Robots-Tag": "noindex, nofollow, noarchive, nosnippet, noimageindex" } });
  }
  const response = await next();
  const protectedResponse = new Response(response.body, response);
  protectedResponse.headers.set("X-Robots-Tag", "noindex, nofollow, noarchive, nosnippet, noimageindex");
  return protectedResponse;
}

