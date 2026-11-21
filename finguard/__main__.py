"""Command-line interface for setting up and using FinGuard."""
import argparse
import json

from . import data, model, tools


def _print(value):
    print(json.dumps(value, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(prog="finguard", description="Fraud and transaction-risk copilot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="build the database and train the fraud models")
    sub.add_parser("train-models", help="train and evaluate the fraud models on the project dataset")
    transaction = sub.add_parser("transaction", help="show a transaction without its labels")
    transaction.add_argument("txn_id", type=int)
    score = sub.add_parser("score", help="score one transaction")
    score.add_argument("txn_id", type=int)
    score.add_argument("--model", choices=model.MODEL_NAMES, default=model.DEFAULT_MODEL)
    query = sub.add_parser("query", help="run a governed read-only SQL query")
    query.add_argument("sql")
    policy = sub.add_parser("policy", help="search the synthetic fraud policies")
    policy.add_argument("query")
    investigate = sub.add_parser("investigate", help="run the LLM investigation agent (uses API credits)")
    investigate.add_argument("txn_id", type=int)
    sub.add_parser("serve", help="serve the HTTP API on localhost:8000")
    mcp_http = sub.add_parser("mcp-http", help="serve MCP over streamable HTTP for ChatGPT or MCP clients")
    mcp_http.add_argument("--host", default="127.0.0.1")
    mcp_http.add_argument("--port", type=int, default=8001)
    mcp_http.add_argument("--allowed-host", action="append", default=[],
                          help="additional exact Host header allowed through a trusted HTTPS proxy")
    args = parser.parse_args()

    if args.command == "init":
        count = data.build_db()
        models = model.train_models()
        _print({"transactions": count, "model": models[model.DEFAULT_MODEL], "models": models})
    elif args.command == "train-models":
        _print(model.train_models())
    elif args.command == "transaction":
        _print(tools.get_transaction(args.txn_id))
    elif args.command == "score":
        _print(tools.score_transaction(args.txn_id, args.model))
    elif args.command == "query":
        _print(tools.run_select(args.sql))
    elif args.command == "policy":
        from .policy import PolicyIndex
        _print(PolicyIndex().search(args.query))
    elif args.command == "investigate":
        from .agent import investigate as run
        _print(run(args.txn_id))
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("finguard.api:app", host="127.0.0.1", port=8000)
    elif args.command == "mcp-http":
        import asyncio
        from mcp.server.transport_security import TransportSecuritySettings
        from .mcp_server import mcp
        allowed_hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", *args.allowed_host]
        allowed_origins = [f"https://{host}" for host in args.allowed_host]
        security = TransportSecuritySettings(allowed_hosts=allowed_hosts, allowed_origins=allowed_origins)
        asyncio.run(mcp.run_streamable_http_async(host=args.host, port=args.port,
                                                   streamable_http_path="/mcp", stateless_http=True,
                                                   transport_security=security))


if __name__ == "__main__":
    main()
