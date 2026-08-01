from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section2Scene(TeachingScene):
    def construct(self):
        # === Setup Layout ===
        title = "Prerequisite Check: What is a Basis?"
        lecture_lines = [
            "A basis is a set of spanning vectors.",
            "Standard unit vectors are i-hat and j-hat.",
            "Every vector is a recipe of basis vectors."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Colors
        COLOR_I = "#00FF00"  # Green
        COLOR_J = "#0000FF"  # Blue
        COLOR_V = "#FFFFFF"  # White
        COLOR_HIGHLIGHT = "#FFFF00"  # Yellow for active line highlighting

        # Origin point for the coordinate system
        origin = self.grid['D3']

        # === Animation for Lecture Line 1 ===
        # "A basis is a set of spanning vectors."
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        # Add visual context: a subtle grid on the right side
        grid_lines = VGroup()
        for r in ["A", "B", "C", "D", "E", "F"]:
            grid_lines.add(Line(self.grid[f"{r}1"], self.grid[f"{r}6"], stroke_width=1, stroke_opacity=0.3, color=GRAY))
        for c in ["1", "2", "3", "4", "5", "6"]:
            grid_lines.add(Line(self.grid[f"A{c}"], self.grid[f"F{c}"], stroke_width=1, stroke_opacity=0.3, color=GRAY))
        
        self.play(Create(grid_lines))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Standard unit vectors are i-hat and j-hat."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # i-hat (1, 0) -> from D3 to D4
        i_vec = Arrow(origin, self.grid['D4'], buff=0, color=COLOR_I, stroke_width=4)
        # i-label at E4 per Issue 34
        i_label = Text("i", slant=ITALIC, color=COLOR_I, font_size=24)
        self.place_at_grid(i_label, 'E4', scale_factor=0.8)
        
        # j-hat (0, 1) -> from D3 to C3
        j_vec = Arrow(origin, self.grid['C3'], buff=0, color=COLOR_J, stroke_width=4)
        j_label = Text("j", slant=ITALIC, color=COLOR_J, font_size=24).next_to(j_vec, LEFT, buff=0.1)
        
        self.play(GrowArrow(i_vec), Write(i_label))
        self.play(GrowArrow(j_vec), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Every vector is a recipe of basis vectors."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Target vector v (3, 2) -> from D3 to B6
        v_vec = Arrow(origin, self.grid['B6'], buff=0, color=COLOR_V, stroke_width=5)
        # v-label at A6 per Issue 33
        v_label = Text("v", slant=ITALIC, color=COLOR_V, font_size=24)
        self.place_at_grid(v_label, 'A6', scale_factor=0.8)
        
        # Recipe icon integration per Issue 26
        recipe_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/recipe.svg")
        self.place_at_grid(recipe_icon, 'B1', scale_factor=0.5)
        recipe_icon.set_color(WHITE)

        self.play(Create(v_vec), Write(v_label), FadeIn(recipe_icon))
        self.wait(1)
        
        # Step-by-step recipe: 3*i then 2*j
        i_steps = VGroup(
            Arrow(origin, self.grid['D4'], buff=0, color=COLOR_I, stroke_width=3),
            Arrow(self.grid['D4'], self.grid['D5'], buff=0, color=COLOR_I, stroke_width=3),
            Arrow(self.grid['D5'], self.grid['D6'], buff=0, color=COLOR_I, stroke_width=3)
        )
        
        j_steps = VGroup(
            Arrow(self.grid['D6'], self.grid['C6'], buff=0, color=COLOR_J, stroke_width=3),
            Arrow(self.grid['C6'], self.grid['B6'], buff=0, color=COLOR_J, stroke_width=3)
        )
        
        # Show the summation equation
        equation = Text("v = 3i + 2j", slant=ITALIC, color=WHITE, font_size=24)
        # Equation in area A3 to A5 per Issue 32
        self.place_in_area(equation, 'A3', 'A5', scale_factor=0.8)
        
        # Animation sequence for the steps
        for step in i_steps:
            self.play(GrowArrow(step), run_time=0.5)
        for step in j_steps:
            self.play(GrowArrow(step), run_time=0.5)
            
        self.play(Write(equation))
        self.wait(2)
