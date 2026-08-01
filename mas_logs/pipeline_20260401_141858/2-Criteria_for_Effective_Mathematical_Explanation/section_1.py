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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout
        lecture_lines = [
            "Imagine an architect giving vague instructions.",
            "Pip is confused by 'steepness-ish' for his slide.",
            "Without precision, logical structures fall apart."
        ]
        self.setup_layout("The Hook: The 'Confused Architect' Scenario", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Architect Asset
        architect = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/architect.svg")
        self.place_in_area(architect, "A1", "C2", scale_factor=0.8)
        
        # Table and Messy Scribble (Blueprint)
        table = Rectangle(height=1, width=2, color=GREY_B).set_fill(GREY_E, opacity=1)
        self.place_at_grid(table, "C2", scale_factor=1.0)
        
        # Create a "messy scribble" using multiple small random lines
        scribble_lines = []
        for _ in range(15):
            start = [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.2, 0.2), 0]
            end = [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.2, 0.2), 0]
            scribble_lines.append(Line(start, end, stroke_width=2, color="#A9A9A9"))
        scribble = VGroup(*scribble_lines)
        scribble.move_to(table.get_center())
        
        blueprint_label = Text("Blueprint", font_size=16, color=WHITE)
        blueprint_label.next_to(table, DOWN, buff=0.1)
        
        architect_group = VGroup(architect, table, scribble, blueprint_label)
        self.play(FadeIn(architect_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Reset Line 1, Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE)
        )
        
        # Penguin Pip Asset
        pip = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/penguin.svg")
        self.place_in_area(pip, "D1", "F2", scale_factor=0.8)
        
        # Wavy, dashed line (the slide)
        # Using a sine wave to represent "steepness-ish"
        wavy_func = FunctionGraph(
            lambda x: 0.3 * np.sin(3 * x),
            x_range=[-1.5, 1.5],
            color="#FF0000"
        )
        wavy_slide = DashedVMobject(wavy_func, num_dashes=30)
        self.place_in_area(wavy_slide, "D3", "F5", scale_factor=0.7)
        
        slide_label = Text("steepness-ish?", font_size=18, color=WHITE)
        slide_label.next_to(wavy_slide, UP, buff=0.2)
        
        question_mark = Text("?", font_size=60, color="#FFFFFF")
        self.place_at_grid(question_mark, "E1", scale_factor=1.0)
        
        pip_group = VGroup(pip, wavy_slide, slide_label, question_mark)
        self.play(FadeIn(pip_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset Line 2, Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(RED)
        )
        
        # Logical structures fall apart: Scribble and Wavy line collapse
        # We break the scribble and wavy slide into pieces and scatter them
        
        collapse_animations = []
        
        # Scatter the blueprint scribble
        for line in scribble:
            target = line.get_center() + np.array([np.random.uniform(-1, 1), -2, 0])
            collapse_animations.append(line.animate.move_to(target).rotate(PI/2).set_opacity(0))
            
        # Scatter the wavy slide (by breaking it into dashed segments)
        for dash in wavy_slide:
            target = dash.get_center() + np.array([np.random.uniform(-1, 1), -2, 0])
            collapse_animations.append(dash.animate.move_to(target).set_opacity(0))
            
        # Also move labels and table down/out
        collapse_animations.append(FadeOut(table, shift=DOWN))
        collapse_animations.append(FadeOut(blueprint_label, shift=DOWN))
        collapse_animations.append(FadeOut(slide_label, shift=DOWN))
        collapse_animations.append(question_mark.animate.scale(0.1).set_opacity(0))
        
        self.play(*collapse_animations, run_time=2)
        self.wait(2)
