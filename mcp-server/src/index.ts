import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

// Configuration
const CHROMADB_HOST = process.env.CHROMADB_HOST || "<ADD-IP-ADDRESS>";
const CHROMADB_PORT = process.env.CHROMADB_PORT || "8000";
const OLLAMA_HOST = process.env.OLLAMA_HOST || "<ADD-IP-ADDRESS>";
const OLLAMA_PORT = process.env.OLLAMA_PORT || "11434";
const SCRIPTS_PATH = "/path/to/hybrid-rag-system/scripts";

const server = new Server(
  {
    name: "unraid-mcp-server",
    version: "2.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Helper to run commands with longer timeout
async function runCommand(cmd: string, timeout = 30000): Promise<string> {
  try {
    const { stdout, stderr } = await execAsync(cmd, { timeout });
    return stdout || stderr;
  } catch (error: any) {
    return `Error: ${error.message}`;
  }
}

// Helper for HTTP requests
async function httpRequest(url: string, options: any = {}): Promise<any> {
  const https = await import("http");
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const req = https.request(
      {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: options.method || "GET",
        headers: options.headers || {},
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            resolve(JSON.parse(data));
          } catch {
            resolve(data);
          }
        });
      }
    );
    req.on("error", reject);
    if (options.body) req.write(JSON.stringify(options.body));
    req.end();
  });
}

// Define available tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    // ===== DOCKER TOOLS =====
    {
      name: "docker_list",
      description: "List all Docker containers with their status",
      inputSchema: {
        type: "object",
        properties: {
          all: { type: "boolean", description: "Show all containers (including stopped)", default: true },
        },
      },
    },
    {
      name: "docker_start",
      description: "Start a Docker container",
      inputSchema: {
        type: "object",
        properties: { container: { type: "string", description: "Container name or ID" } },
        required: ["container"],
      },
    },
    {
      name: "docker_stop",
      description: "Stop a Docker container",
      inputSchema: {
        type: "object",
        properties: { container: { type: "string", description: "Container name or ID" } },
        required: ["container"],
      },
    },
    {
      name: "docker_restart",
      description: "Restart a Docker container",
      inputSchema: {
        type: "object",
        properties: { container: { type: "string", description: "Container name or ID" } },
        required: ["container"],
      },
    },
    {
      name: "docker_logs",
      description: "Get logs from a Docker container",
      inputSchema: {
        type: "object",
        properties: {
          container: { type: "string", description: "Container name or ID" },
          lines: { type: "number", description: "Number of log lines", default: 50 },
        },
        required: ["container"],
      },
    },
    {
      name: "docker_stats",
      description: "Get resource usage stats for running containers",
      inputSchema: { type: "object", properties: {} },
    },

    // ===== SYSTEM TOOLS =====
    {
      name: "system_info",
      description: "Get Unraid system information (CPU, RAM, uptime)",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "disk_usage",
      description: "Get disk usage for all mounts",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "array_status",
      description: "Get Unraid array status and disk health",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "network_info",
      description: "Get network interface information",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "shares_list",
      description: "List Unraid shares",
      inputSchema: { type: "object", properties: {} },
    },

    // ===== CHROMADB TOOLS =====
    {
      name: "chromadb_query",
      description: "Search the ChromaDB knowledge base for relevant documentation",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          n_results: { type: "number", description: "Number of results", default: 5 },
          collection: { type: "string", description: "Collection name", default: "infrastructure_docs" },
        },
        required: ["query"],
      },
    },
    {
      name: "chromadb_add",
      description: "Add a document to the ChromaDB knowledge base",
      inputSchema: {
        type: "object",
        properties: {
          document: { type: "string", description: "Document content to add" },
          metadata: { type: "object", description: "Optional metadata (source, category, etc.)" },
          collection: { type: "string", description: "Collection name", default: "infrastructure_docs" },
        },
        required: ["document"],
      },
    },
    {
      name: "chromadb_collections",
      description: "List all ChromaDB collections",
      inputSchema: { type: "object", properties: {} },
    },

    // ===== OLLAMA TOOLS =====
    {
      name: "ollama_query",
      description: "Query the local Ollama LLM directly",
      inputSchema: {
        type: "object",
        properties: {
          prompt: { type: "string", description: "Prompt to send to Ollama" },
          model: { type: "string", description: "Model to use", default: "mistral" },
          context: { type: "string", description: "Optional context to include" },
        },
        required: ["prompt"],
      },
    },
    {
      name: "ollama_models",
      description: "List available Ollama models",
      inputSchema: { type: "object", properties: {} },
    },

    // ===== FULL TOOLSET INTEGRATION =====
    {
      name: "ask_local_ai",
      description: "Ask the local AI system using the full toolset (can use SSH, file ops, web search, etc.)",
      inputSchema: {
        type: "object",
        properties: {
          question: { type: "string", description: "Question to ask the local AI" },
          mode: {
            type: "string",
            enum: ["fulltools", "hybrid", "rag"],
            description: "Mode: fulltools (12 tools), hybrid (RAG + Claude fallback), rag (basic RAG)",
            default: "fulltools",
          },
        },
        required: ["question"],
      },
    },
    {
      name: "index_knowledge",
      description: "Update the ChromaDB knowledge base by indexing documentation from all systems",
      inputSchema: {
        type: "object",
        properties: {
          system: {
            type: "string",
            enum: ["all", "local", "proxmox", "unraid"],
            description: "Which system to index",
            default: "all",
          },
        },
      },
    },

    // ===== GENERAL =====
    {
      name: "run_command",
      description: "Run a custom shell command on Unraid",
      inputSchema: {
        type: "object",
        properties: { command: { type: "string", description: "Command to execute" } },
        required: ["command"],
      },
    },
  ],
}));

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    // ===== DOCKER TOOLS =====
    case "docker_list": {
      const all = (args as any)?.all !== false ? "-a" : "";
      const result = await runCommand(
        `docker ps ${all} --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"`
      );
      return { content: [{ type: "text", text: result }] };
    }

    case "docker_start": {
      const result = await runCommand(`docker start ${(args as any).container}`);
      return { content: [{ type: "text", text: `Started: ${result}` }] };
    }

    case "docker_stop": {
      const result = await runCommand(`docker stop ${(args as any).container}`);
      return { content: [{ type: "text", text: `Stopped: ${result}` }] };
    }

    case "docker_restart": {
      const result = await runCommand(`docker restart ${(args as any).container}`);
      return { content: [{ type: "text", text: `Restarted: ${result}` }] };
    }

    case "docker_logs": {
      const { container, lines = 50 } = args as any;
      const result = await runCommand(`docker logs --tail ${lines} ${container}`);
      return { content: [{ type: "text", text: result }] };
    }

    case "docker_stats": {
      const result = await runCommand(
        `docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"`
      );
      return { content: [{ type: "text", text: result }] };
    }

    // ===== SYSTEM TOOLS =====
    case "system_info": {
      const cpu = await runCommand(`grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2`);
      const cores = await runCommand(`nproc`);
      const load = await runCommand(`cat /proc/loadavg`);
      const memory = await runCommand(`free -h | grep Mem`);
      const uptime = await runCommand(`uptime -p`);
      return {
        content: [{
          type: "text",
          text: `=== Unraid System Info ===\nCPU:${cpu.trim()}\nCores: ${cores.trim()}\nLoad: ${load.trim()}\nMemory: ${memory.trim()}\nUptime: ${uptime.trim()}`,
        }],
      };
    }

    case "disk_usage": {
      const result = await runCommand(`df -h | grep -E "^/dev|mnt"`);
      return { content: [{ type: "text", text: result }] };
    }

    case "array_status": {
      const mdstat = await runCommand(`cat /proc/mdstat 2>/dev/null || echo "N/A"`);
      const cacheInfo = await runCommand(`df -h /mnt/cache 2>/dev/null || echo "No cache"`);
      return {
        content: [{ type: "text", text: `=== Array Status ===\n${mdstat}\n\nCache:\n${cacheInfo}` }],
      };
    }

    case "network_info": {
      const result = await runCommand(`ip -brief addr show`);
      return { content: [{ type: "text", text: result }] };
    }

    case "shares_list": {
      const result = await runCommand(`ls -la /mnt/user/`);
      return { content: [{ type: "text", text: result }] };
    }

    // ===== CHROMADB TOOLS =====
    case "chromadb_query": {
      const { query, n_results = 5, collection = "infrastructure_docs" } = args as any;
      try {
        const result = await runCommand(
          `curl -s -X POST "http://${CHROMADB_HOST}:${CHROMADB_PORT}/api/v2/collections/${collection}/query" ` +
          `-H "Content-Type: application/json" ` +
          `-d '{"query_texts": ["${query.replace(/"/g, '\\"')}"], "n_results": ${n_results}, "include": ["documents", "metadatas", "distances"]}'`,
          60000
        );
        const data = JSON.parse(result);
        if (data.documents && data.documents[0]) {
          let output = `Found ${data.documents[0].length} results:\n\n`;
          data.documents[0].forEach((doc: string, i: number) => {
            const meta = data.metadatas?.[0]?.[i] || {};
            const dist = data.distances?.[0]?.[i] || "N/A";
            output += `--- Result ${i + 1} (distance: ${typeof dist === 'number' ? dist.toFixed(3) : dist}) ---\n`;
            output += `Source: ${meta.source || "unknown"}\n`;
            output += `${doc.substring(0, 500)}${doc.length > 500 ? "..." : ""}\n\n`;
          });
          return { content: [{ type: "text", text: output }] };
        }
        return { content: [{ type: "text", text: "No results found" }] };
      } catch (e: any) {
        return { content: [{ type: "text", text: `ChromaDB query failed: ${e.message}` }] };
      }
    }

    case "chromadb_add": {
      const { document, metadata = {}, collection = "infrastructure_docs" } = args as any;
      const docId = `mcp-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
      try {
        const result = await runCommand(
          `curl -s -X POST "http://${CHROMADB_HOST}:${CHROMADB_PORT}/api/v2/collections/${collection}/add" ` +
          `-H "Content-Type: application/json" ` +
          `-d '{"ids": ["${docId}"], "documents": ["${document.replace(/"/g, '\\"').replace(/\n/g, "\\n")}"], "metadatas": [${JSON.stringify({ ...metadata, added_via: "mcp", added_at: new Date().toISOString() })}]}'`
        );
        return { content: [{ type: "text", text: `Added document with ID: ${docId}` }] };
      } catch (e: any) {
        return { content: [{ type: "text", text: `Failed to add: ${e.message}` }] };
      }
    }

    case "chromadb_collections": {
      const result = await runCommand(
        `curl -s "http://${CHROMADB_HOST}:${CHROMADB_PORT}/api/v2/collections"`
      );
      return { content: [{ type: "text", text: result }] };
    }

    // ===== OLLAMA TOOLS =====
    case "ollama_query": {
      const { prompt, model = "mistral", context = "" } = args as any;
      const fullPrompt = context ? `Context:\n${context}\n\nQuestion: ${prompt}` : prompt;
      try {
        const result = await runCommand(
          `curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/generate" ` +
          `-d '{"model": "${model}", "prompt": "${fullPrompt.replace(/"/g, '\\"').replace(/\n/g, "\\n")}", "stream": false}'`,
          120000
        );
        const data = JSON.parse(result);
        return { content: [{ type: "text", text: data.response || "No response" }] };
      } catch (e: any) {
        return { content: [{ type: "text", text: `Ollama query failed: ${e.message}` }] };
      }
    }

    case "ollama_models": {
      const result = await runCommand(`curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags"`);
      try {
        const data = JSON.parse(result);
        const models = data.models?.map((m: any) => `${m.name} (${(m.size / 1e9).toFixed(1)}GB)`).join("\n") || "No models";
        return { content: [{ type: "text", text: `Available models:\n${models}` }] };
      } catch {
        return { content: [{ type: "text", text: result }] };
      }
    }

    // ===== FULL TOOLSET INTEGRATION =====
    case "ask_local_ai": {
      const { question, mode = "fulltools" } = args as any;
      const script = mode === "hybrid" ? "hybrid-rag.py" : mode === "rag" ? "rag-demo.py" : "ollama-full-toolset.py";
      const result = await runCommand(
        `cd ${SCRIPTS_PATH} && python3 ${script} "${question.replace(/"/g, '\\"')}"`,
        300000 // 5 min timeout for full toolset
      );
      return { content: [{ type: "text", text: result }] };
    }

    case "index_knowledge": {
      const { system = "all" } = args as any;
      const flag = system === "all" ? "" : `--${system}-only`;
      const result = await runCommand(
        `cd ${SCRIPTS_PATH} && python3 index-all-systems.py ${flag}`,
        600000 // 10 min timeout
      );
      return { content: [{ type: "text", text: result }] };
    }

    // ===== GENERAL =====
    case "run_command": {
      const { command } = args as any;
      const dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){"];
      if (dangerous.some((d) => command.toLowerCase().includes(d))) {
        return { content: [{ type: "text", text: "Command blocked for safety" }] };
      }
      const result = await runCommand(command);
      return { content: [{ type: "text", text: result }] };
    }

    default:
      return { content: [{ type: "text", text: `Unknown tool: ${name}` }] };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Unraid MCP Server v2.0 running (Docker, ChromaDB, Ollama, Full Toolset)");
}

main().catch(console.error);
