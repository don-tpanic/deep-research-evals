from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import re

"""
Placeholder code (auto-generated) for building evaluation rubrics for AI agents.
"""

class PerformanceLevel(Enum):
    EXCELLENT = 4
    GOOD = 3
    SATISFACTORY = 2
    NEEDS_IMPROVEMENT = 1
    POOR = 0


@dataclass
class RubricComponent:
    """Represents a single evaluation component in the rubric."""
    name: str
    weight: float  # As percentage (0-100)
    description: str
    performance_levels: Dict[PerformanceLevel, str]


@dataclass
class EvaluationRubric:
    """Complete evaluation rubric with all components."""
    title: str
    components: List[RubricComponent]
    overall_guidance: str
    scoring_method: str


@dataclass
class AgentPrompt:
    """The original prompt given to the agent."""
    title: str
    content: str
    requirements: List[str]  # Bullet points or requirements extracted


@dataclass
class AgentTrajectory:
    """Optional: The agent's actual response to evaluate against."""
    prompt_used: str
    agent_response: str
    timestamp: Optional[str] = None


class LLMClient:
    """Simple interface for LLM API calls."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Mock implementation - replace with actual API call
        (Anthropic, OpenAI, etc.)
        """
        # This would be replaced with actual API call
        # For now, return placeholder
        return "Generated rubric content would go here"


class RubricGenerator:
    """Main class that generates evaluation rubrics from prompts."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def extract_requirements(self, prompt_content: str) -> List[str]:
        """Extract bullet points and requirements from prompt."""
        # Simple regex to find bullet points
        bullet_patterns = [
            r'^\s*[-*•]\s*(.+)$',  # - * • bullets
            r'^\s*\d+\.\s*(.+)$',  # numbered lists
            r'^\s*[a-zA-Z]\.\s*(.+)$'  # lettered lists
        ]
        
        requirements = []
        lines = prompt_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in bullet_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    requirements.append(match.group(1).strip())
                    break
        
        return requirements
    
    def generate_rubric(
        self, 
        agent_prompt: AgentPrompt,
        trajectory: Optional[AgentTrajectory] = None,
        domain_context: Optional[str] = None
    ) -> EvaluationRubric:
        """Generate evaluation rubric from agent prompt."""
        
        # Build the generation prompt
        generation_prompt = self._build_generation_prompt(
            agent_prompt, trajectory, domain_context
        )
        
        # Call LLM to generate rubric
        rubric_response = self.llm_client.generate(generation_prompt)
        
        # Parse the response into structured rubric
        rubric = self._parse_rubric_response(rubric_response, agent_prompt.title)
        
        return rubric
    
    def _build_generation_prompt(
        self,
        agent_prompt: AgentPrompt,
        trajectory: Optional[AgentTrajectory],
        domain_context: Optional[str]
    ) -> str:
        """Build the prompt for LLM to generate rubric."""
        
        base_prompt = f"""
You are an expert in creating evaluation rubrics for AI agents. Your task is to create a comprehensive evaluation rubric based on the given agent prompt.

AGENT PROMPT TO EVALUATE:
Title: {agent_prompt.title}
Content: {agent_prompt.content}

EXTRACTED REQUIREMENTS:
{chr(10).join(f"- {req}" for req in agent_prompt.requirements)}

Your task is to create an evaluation rubric that assesses how well an agent follows these requirements.

RUBRIC GENERATION INSTRUCTIONS:
1. **Identify 4-6 key evaluation components** from the requirements
2. **Assign weights** to each component (totaling 100%)
3. **Create 5 performance levels** (0-4 scale: Poor, Needs Improvement, Satisfactory, Good, Excellent)
4. **Write specific descriptors** for each performance level
5. **Include overall scoring guidance**

RUBRIC FORMAT:
Please structure your response as follows:

# [Rubric Title]

## Component 1: [Name] (Weight: X%)
**Excellent (4)**: [Specific high-performance behaviors]
**Good (3)**: [Solid performance with minor gaps]  
**Satisfactory (2)**: [Basic competency shown]
**Needs Improvement (1)**: [Significant deficiencies]
**Poor (0)**: [No evidence of capability]

[Repeat for all components]

## Overall Scoring Guidance
[Instructions on how to combine component scores]

## Quality Indicators
**Look for evidence of:**
- [Key quality markers]

## Red Flags  
**Deduct points for:**
- [Common failure modes]
"""

        if domain_context:
            base_prompt += f"\n\nDOMAIN CONTEXT: {domain_context}"
        
        if trajectory:
            base_prompt += f"""
            
EXAMPLE AGENT RESPONSE (for context):
{trajectory.agent_response[:1000]}...
[Use this to understand what kinds of responses need evaluation, but don't evaluate this specific response]
"""
        
        return base_prompt
    
    def _parse_rubric_response(self, response: str, title: str) -> EvaluationRubric:
        """Parse LLM response into structured rubric object."""
        # This is a simplified parser - in practice you'd want more robust parsing
        
        components = []
        
        # Extract components using regex (simplified)
        component_pattern = r'## (.+?): (.+?) \(Weight: (\d+)%\)(.*?)(?=##|$)'
        matches = re.findall(component_pattern, response, re.DOTALL)
        
        for match in matches:
            component_name = match[1].strip()
            weight = float(match[2])
            content = match[3]
            
            # Extract performance levels
            performance_levels = {}
            level_pattern = r'\*\*(\w+) \((\d+)\)\*\*:\s*(.+?)(?=\*\*|\n\n|$)'
            level_matches = re.findall(level_pattern, content, re.DOTALL)
            
            for level_match in level_matches:
                level_name = level_match[0].upper()
                level_score = int(level_match[1])
                level_desc = level_match[2].strip()
                
                if level_name == "EXCELLENT":
                    performance_levels[PerformanceLevel.EXCELLENT] = level_desc
                elif level_name == "GOOD":
                    performance_levels[PerformanceLevel.GOOD] = level_desc
                elif level_name == "SATISFACTORY":
                    performance_levels[PerformanceLevel.SATISFACTORY] = level_desc
                elif level_name.startswith("NEED"):
                    performance_levels[PerformanceLevel.NEEDS_IMPROVEMENT] = level_desc
                elif level_name == "POOR":
                    performance_levels[PerformanceLevel.POOR] = level_desc
            
            component = RubricComponent(
                name=component_name,
                weight=weight,
                description="",
                performance_levels=performance_levels
            )
            components.append(component)
        
        return EvaluationRubric(
            title=f"Evaluation Rubric: {title}",
            components=components,
            overall_guidance="Extracted from LLM response",
            scoring_method="Weighted component scoring (0-4 scale)"
        )
    
    def export_rubric(self, rubric: EvaluationRubric, format: str = "json") -> str:
        """Export rubric in specified format."""
        if format == "json":
            return self._export_json(rubric)
        elif format == "markdown":
            return self._export_markdown(rubric)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, rubric: EvaluationRubric) -> str:
        """Export rubric as JSON."""
        rubric_dict = {
            "title": rubric.title,
            "scoring_method": rubric.scoring_method,
            "components": []
        }
        
        for component in rubric.components:
            component_dict = {
                "name": component.name,
                "weight": component.weight,
                "description": component.description,
                "performance_levels": {
                    level.name.lower(): desc 
                    for level, desc in component.performance_levels.items()
                }
            }
            rubric_dict["components"].append(component_dict)
        
        return json.dumps(rubric_dict, indent=2)
    
    def _export_markdown(self, rubric: EvaluationRubric) -> str:
        """Export rubric as Markdown."""
        md_content = f"# {rubric.title}\n\n"
        
        for component in rubric.components:
            md_content += f"## {component.name} (Weight: {component.weight}%)\n\n"
            
            for level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD, 
                         PerformanceLevel.SATISFACTORY, PerformanceLevel.NEEDS_IMPROVEMENT, 
                         PerformanceLevel.POOR]:
                if level in component.performance_levels:
                    level_name = level.name.replace('_', ' ').title()
                    md_content += f"**{level_name} ({level.value})**: {component.performance_levels[level]}\n\n"
        
        return md_content


def main():
    """Example usage of the rubric generator."""
    
    # Initialize LLM client (replace with actual implementation)
    llm_client = LLMClient(api_key="your-api-key")
    generator = RubricGenerator(llm_client)
    
    # Define the agent prompt
    prompt = AgentPrompt(
        title="Assessment and Breakdown",
        content="""
        1. **Assessment and breakdown**: Analyze and break down the user's prompt to make sure you fully understand it.
        * Identify the main concepts, key entities, and relationships in the task.
        * List specific facts or data points needed to answer the question well.
        * Note any temporal or contextual constraints on the question.
        * Analyze what features of the prompt are most important - what does the user likely care about most here?
        * Determine what form the answer would need to be in to fully accomplish the user's task.
        """,
        requirements=[
            "Identify the main concepts, key entities, and relationships in the task",
            "List specific facts or data points needed to answer the question well",
            "Note any temporal or contextual constraints on the question",
            "Analyze what features of the prompt are most important",
            "Determine what form the answer would need to be in"
        ]
    )
    
    # Optional: Include agent trajectory
    trajectory = AgentTrajectory(
        prompt_used=prompt.content,
        agent_response="Agent's actual response would go here..."
    )
    
    # Generate rubric
    rubric = generator.generate_rubric(
        agent_prompt=prompt,
        trajectory=trajectory,
        domain_context="Research and analysis tasks"
    )
    
    # Export rubric
    json_output = generator.export_rubric(rubric, "json")
    markdown_output = generator.export_rubric(rubric, "markdown")
    
    print("Generated Rubric (JSON):")
    print(json_output)
    print("\nGenerated Rubric (Markdown):")
    print(markdown_output)


if __name__ == "__main__":
    main()