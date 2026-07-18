import json
import re

from kb_config import get_neo4j_driver, get_gemini_model

EXTRACTION_PROMPT = """Analyze the following text and extract all entities and relationships.

Entity types to look for: Person, Department, Role, Policy, Document, Project, Date, Organization, Location, Skill, Process.

Return ONLY valid JSON in this exact format (no markdown, no code fences):
{
  "entities": [
    {"name": "entity name", "type": "entity type", "properties": {"key": "value"}}
  ],
  "relationships": [
    {"source": "source entity name", "target": "target entity name", "type": "relationship_type", "properties": {"key": "value"}}
  ]
}

Relationship types: REPORTS_TO, BELONGS_TO, AUTHORED_BY, MANAGES, WORKS_ON, HAS_SKILL, FOLLOWS_POLICY, RELATED_TO, CREATED_ON, LOCATED_IN, PART_OF.

Text to analyze:
---
{text}
---

Extract all entities and relationships you can identify. Be thorough."""

ENTITY_EXTRACTION_PROMPT = """From the following query, extract the key entities (names, departments, roles, policies, projects, etc.) that should be searched in a knowledge graph.

Return ONLY valid JSON (no markdown, no code fences):
{{"entities": ["entity1", "entity2", ...]}}

Query: {query}"""


def extract_entities_and_relationships(text: str) -> dict:
    model = get_gemini_model()
    prompt = EXTRACTION_PROMPT.format(text=text[:8000])

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        if "entities" not in result:
            result["entities"] = []
        if "relationships" not in result:
            result["relationships"] = []
        return result
    except (json.JSONDecodeError, Exception) as e:
        print(f"Entity extraction failed: {e}")
        return {"entities": [], "relationships": []}


def store_in_neo4j(entities: list, relationships: list, source_doc: str):
    driver = get_neo4j_driver()

    with driver.session() as session:
        for entity in entities:
            name = entity.get("name", "").strip()
            etype = entity.get("type", "Entity").strip()
            props = entity.get("properties", {})
            if not name:
                continue

            session.run(
                """
                MERGE (n:Entity {name: $name, type: $type})
                SET n.source_doc = $source_doc
                SET n += $props
                """,
                name=name,
                type=etype,
                source_doc=source_doc,
                props=props,
            )

        for rel in relationships:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            rel_type = rel.get("type", "RELATED_TO").strip().upper()
            props = rel.get("properties", {})
            if not source or not target:
                continue

            rel_type_clean = re.sub(r"[^A-Z0-9_]", "_", rel_type)

            session.run(
                f"""
                MATCH (a:Entity {{name: $source}})
                MATCH (b:Entity {{name: $target}})
                MERGE (a)-[r:{rel_type_clean}]->(b)
                SET r.source_doc = $source_doc
                SET r += $props
                """,
                source=source,
                target=target,
                source_doc=source_doc,
                props=props,
            )

    driver.close()


def extract_query_entities(query: str) -> list:
    model = get_gemini_model()
    prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        return result.get("entities", [])
    except Exception:
        words = query.split()
        return [w for w in words if len(w) > 3 and w[0].isupper()]


def query_knowledge_graph(query: str) -> list:
    entities = extract_query_entities(query)
    if not entities:
        return []

    driver = get_neo4j_driver()
    triples = []

    with driver.session() as session:
        for entity in entities:
            result = session.run(
                """
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($entity)
                OPTIONAL MATCH (n)-[r]->(m:Entity)
                RETURN n.name AS source, n.type AS source_type,
                       type(r) AS relationship, m.name AS target,
                       m.type AS target_type, n.source_doc AS source_doc
                LIMIT 20
                """,
                entity=entity,
            )
            for record in result:
                if record["relationship"] and record["target"]:
                    triples.append({
                        "source": record["source"],
                        "source_type": record["source_type"],
                        "relationship": record["relationship"],
                        "target": record["target"],
                        "target_type": record["target_type"],
                        "source_doc": record["source_doc"],
                    })

            result_incoming = session.run(
                """
                MATCH (m:Entity)-[r]->(n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($entity)
                RETURN m.name AS source, m.type AS source_type,
                       type(r) AS relationship, n.name AS target,
                       n.type AS target_type, n.source_doc AS source_doc
                LIMIT 20
                """,
                entity=entity,
            )
            for record in result_incoming:
                if record["relationship"]:
                    triples.append({
                        "source": record["source"],
                        "source_type": record["source_type"],
                        "relationship": record["relationship"],
                        "target": record["target"],
                        "target_type": record["target_type"],
                        "source_doc": record["source_doc"],
                    })

    driver.close()

    seen = set()
    unique_triples = []
    for t in triples:
        key = (t["source"], t["relationship"], t["target"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    return unique_triples


def format_graph_context(triples: list) -> str:
    if not triples:
        return ""
    lines = []
    for t in triples:
        rel_readable = t["relationship"].replace("_", " ").lower()
        lines.append(
            f"{t['source']} ({t['source_type']}) {rel_readable} {t['target']} ({t['target_type']})"
        )
    return "\n".join(lines)
