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
        lecture_lines = [
            "Meet Pip, a squirrel who loves organizing his acorns.",
            "Three acorns can be arranged in six different ways.",
            "But how many ways can Pip arrange zero acorns?"
        ]
        self.setup_layout("The Hook: The Squirrel's Dilemma", lecture_lines)

        # Colors
        PIP_COLOR = "#FFD700"  
        ACORN_COLOR = "#8B4513" 
        TEXT_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"

        # Assets - Replaced SVGMobjects with internal VGroups to fix OSError
        # [Asset: pip.svg]
        pip = VGroup(
            Ellipse(width=0.6, height=0.8, color=PIP_COLOR, fill_opacity=1),
            Circle(radius=0.2, color=PIP_COLOR, fill_opacity=1).shift(0.5 * UP),
            Triangle(color=PIP_COLOR, fill_opacity=1).scale(0.15).shift(0.4 * UP + 0.25 * RIGHT),
            Triangle(color=PIP_COLOR, fill_opacity=1).scale(0.15).shift(0.4 * UP + 0.25 * LEFT)
        )
        
        # [Asset: acorn.svg]
        def create_acorn():
            return VGroup(
                Ellipse(width=0.35, height=0.45, color=ACORN_COLOR, fill_opacity=1),
                Arc(radius=0.18, angle=PI, color="#5D2E0A", fill_opacity=1).shift(0.15 * UP),
                Line(0.15 * UP, 0.25 * UP, color="#5D2E0A", stroke_width=2)
            )

        acorns = VGroup(create_acorn(), create_acorn(), create_acorn())
        
        # Shelf
        shelf = Line(LEFT, RIGHT, color=GREY).scale(2)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.place_in_area(pip, "B1", "E2", scale_factor=0.8)
        self.place_in_area(shelf, "E3", "E6", scale_factor=1.0)
        
        # Initial acorn positions on shelf
        for i, acorn in enumerate(acorns):
            self.place_at_grid(acorn, f"D{i+4}", scale_factor=0.6)

        self.play(FadeIn(pip), Create(shelf), FadeIn(acorns))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        label_3factorial = Text("3! = 6", color=TEXT_COLOR)
        self.place_at_grid(label_3factorial, "A4")

        self.play(Write(label_3factorial))

        # Permutation visualization (quick cycling)
        positions = [self.grid["D4"], self.grid["D5"], self.grid["D6"]]
        perms = [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        
        for p in perms:
            self.play(
                acorns[0].animate.move_to(positions[p[0]]),
                acorns[1].animate.move_to(positions[p[1]]),
                acorns[2].animate.move_to(positions[p[2]]),
                run_time=0.3
            )
        self.wait(0.5)

        # Content suggests showing 2! and 1! as well
        label_2factorial = Text("2! = 2", color=TEXT_COLOR)
        self.place_at_grid(label_2factorial, "A4") 
        
        label_1factorial = Text("1! = 1", color=TEXT_COLOR)
        self.place_at_grid(label_1factorial, "A4") 

        # Transition to 2 acorns
        self.play(FadeOut(acorns[2]), ReplacementTransform(label_3factorial, label_2factorial))
        self.play(acorns[0].animate.move_to(self.grid["D4"]), acorns[1].animate.move_to(self.grid["D5"]), run_time=0.4)
        self.play(acorns[0].animate.move_to(self.grid["D5"]), acorns[1].animate.move_to(self.grid["D4"]), run_time=0.4)
        
        # Transition to 1 acorn
        self.play(FadeOut(acorns[1]), ReplacementTransform(label_2factorial, label_1factorial))
        self.play(acorns[0].animate.move_to(self.grid["D4"]), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        thought_bubble = Circle(radius=0.8, color=WHITE).scale(0.8)
        self.place_at_grid(thought_bubble, "A3") 
        thought_text = Text("0! = ?", color=TEXT_COLOR)
        self.place_at_grid(thought_text, "A3", scale_factor=0.8) 

        giant_one = Text("1", font_size=120, color=HIGHLIGHT_COLOR)
        self.place_in_area(giant_one, "B4", "D5") 

        self.play(FadeOut(acorns[0]), FadeOut(label_1factorial))
        self.play(Create(thought_bubble), Write(thought_text))
        self.wait(1)

        self.play(FadeIn(giant_one, shift=UP))
        self.play(Indicate(giant_one))
        self.wait(2)
