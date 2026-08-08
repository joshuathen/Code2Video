from manim import *
import numpy as np

# Custom Colors
CYAN_A = "#00FFFF"
WAREHOUSE_BLUE = "#ADD8E6"
PARIS_MAGENTA = "#FF00FF"

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
        lecture_lines = [
            "How do AI models remember facts without internet?",
            "Large Language Models store information in their weights.",
            "We'll explore the Feed-Forward Network as a warehouse.",
            "Meet Lex, a robot looking deep into his circuitry.",
            "He finds \"Paris\" stored within his internal parameters."
        ]
        self.setup_layout("The Mystery of Model Memory", lecture_lines)

        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/warehouse.svg]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        lex = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        lex.set_color(WHITE)
        
        # Fix for Issue 48: Position Lex in area C3 to F4
        self.place_in_area(lex, 'C3', 'F4', scale_factor=0.8)
        
        self.play(DrawBorderThenFill(lex), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(CYAN_A)
        
        # Thought bubble with question mark
        bubble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg") # Fallback if bubble not asset, but instruction says make one
        # Let's create a manual one as per storyboard "Animate a thought bubble with a question mark"
        thought_bubble = VGroup(
            Circle(radius=0.1, color=CYAN_A),
            Circle(radius=0.2, color=CYAN_A).shift(UP*0.3 + RIGHT*0.2),
            RoundedRectangle(corner_radius=0.3, height=1.0, width=1.5, color=CYAN_A).shift(UP*1.0 + RIGHT*0.5)
        )
        q_mark = Text("?", font_size=36, color=CYAN_A).move_to(thought_bubble[2].get_center())
        bubble_group = VGroup(thought_bubble, q_mark)
        
        # Fix for Issue 48: Align bubble with head (C5)
        self.place_at_grid(bubble_group, 'C5', scale_factor=0.8)
        
        self.play(FadeIn(bubble_group, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        
        # Pulse weights inside Lex
        # Create a small network overlay
        weights_overlay = VGroup(*[
            Line(ORIGIN, 0.2*RIGHT, color=YELLOW, stroke_width=2) for _ in range(5)
        ]).arrange_in_grid(2, 3, buff=0.1).move_to(lex.get_center())
        
        self.play(
            FadeIn(weights_overlay),
            weights_overlay.animate.scale(1.2).set_color(WHITE),
            rate_func=there_and_back,
            run_time=1.0
        )
        self.play(FadeOut(weights_overlay))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WAREHOUSE_BLUE)
        
        # Transition Lex/Bubble to Warehouse
        warehouse = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/warehouse.svg")
        warehouse.set_color(WAREHOUSE_BLUE)
        
        # Fix for Issue 48: network/warehouse at D3
        self.place_at_grid(warehouse, 'D3', scale_factor=1.2)
        
        self.play(
            FadeOut(lex),
            FadeOut(bubble_group),
            FadeIn(warehouse, shift=DOWN)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PARIS_MAGENTA)
        
        # Highlight specific weight cell labeled 'Paris'
        answer_label = Text("Paris", color=PARIS_MAGENTA, font_size=32)
        # Fix for Issue 48: Paris label at D5
        self.place_at_grid(answer_label, 'D5', scale_factor=1.0)
        
        highlight_box = SurroundingRectangle(answer_label, color=PARIS_MAGENTA, buff=0.1)
        
        self.play(
            Write(answer_label),
            Create(highlight_box)
        )
        
        # Pulse effect for emphasis
        self.play(
            answer_label.animate.scale(1.2),
            highlight_box.animate.scale(1.1).set_stroke(width=6),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(3)
