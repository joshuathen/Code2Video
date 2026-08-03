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
        # Data from storyboard
        title = "The Fragmented Kingdom: The Problem with Current Notation"
        lines = [
            "Exponents, roots, and logarithms often feel like separate languages.",
            "But these three operations are actually deeply connected siblings.",
            "Our current notation hides this beautiful mathematical symmetry."
        ]
        # Adding bullet points as required by setup_layout convention
        bullet_lines = [f"- {line}" for line in lines]
        self.setup_layout(title, bullet_lines)
        
        # Colors from storyboard
        color_exp = "#ADD8E6"  # Light Blue
        color_root = "#90EE90" # Light Green
        color_log = "#FFB6C1"  # Light Pink
        color_leo = "#FFFFE0"  # Light Yellow
        color_puzzle = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Show '2³=8', '∛8=2', and 'log₂8=3' scattered
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        exp_eq = MathTex("2^3=8", color=color_exp)
        root_eq = MathTex(r"\sqrt[3]{8}=2", color=color_root)
        log_eq = MathTex(r"\log_2 8=3", color=color_log)
        
        self.place_at_grid(exp_eq, "B2", scale_factor=1.0)
        # Resolved Issue 24: Move root_eq to D2
        self.place_at_grid(root_eq, "D2", scale_factor=1.0)
        self.place_at_grid(log_eq, "B5", scale_factor=1.0)
        
        self.play(FadeIn(exp_eq), FadeIn(root_eq), FadeIn(log_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight each expression in the center while others fade
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Define a center for highlighting
        center_mobj = VMobject()
        # Resolved Issue 23: Shift highlight area to B4-E6 to avoid obstructing lecture text
        self.place_in_area(center_mobj, 'B4', 'E6', scale_factor=0.8)
        center_pos = center_mobj.get_center()
        
        # Highlight exp
        self.play(
            exp_eq.animate.move_to(center_pos).scale(1.5),
            root_eq.animate.set_opacity(0.3),
            log_eq.animate.set_opacity(0.3)
        )
        self.wait(1)
        
        # Highlight root
        self.play(
            exp_eq.animate.move_to(self.grid["B2"]).scale(1/1.5).set_opacity(0.3),
            root_eq.animate.move_to(center_pos).scale(1.5).set_opacity(1),
        )
        self.wait(1)
        
        # Highlight log
        self.play(
            root_eq.animate.move_to(self.grid["D2"]).scale(1/1.5).set_opacity(0.3),
            log_eq.animate.move_to(center_pos).scale(1.5).set_opacity(1),
        )
        self.wait(1)
        
        # Return log to its place
        self.play(
            log_eq.animate.move_to(self.grid["B5"]).scale(1/1.5).set_opacity(0.3)
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Character 'Leo' and puzzle boxes
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            FadeOut(exp_eq), FadeOut(root_eq), FadeOut(log_eq)
        )
        
        # Leo character (represented as a yellow circle)
        leo = Circle(radius=0.5, color=color_leo, fill_opacity=1)
        leo_label = Text("Leo", font_size=18, color=color_leo)
        # Resolved Issue 25: Move leo to D4
        self.place_at_grid(leo, "D4", scale_factor=1.0)
        leo_label.next_to(leo, DOWN, buff=0.1)
        
        # Puzzle boxes (white squares)
        box1 = Square(side_length=0.7, color=color_puzzle)
        box2 = Square(side_length=0.7, color=color_puzzle)
        box3 = Square(side_length=0.7, color=color_puzzle)
        
        # Positioning boxes away from leo/question marks to reduce clutter
        self.place_at_grid(box1, "B4")
        self.place_at_grid(box2, "C5")
        self.place_at_grid(box3, "F5") # Moved slightly from E4 to avoid overlap with q3
        
        self.play(FadeIn(leo), FadeIn(leo_label))
        self.play(
            LaggedStart(
                Create(box1), Create(box2), Create(box3),
                lag_ratio=0.3
            )
        )
        
        # Leo's confusion (question marks)
        q1 = Text("?", color=color_leo).scale(0.8)
        q2 = Text("?", color=color_leo).scale(0.8)
        q3 = Text("?", color=color_leo).scale(0.8)
        # Resolved Issue 25: Grid positions for question marks
        self.place_at_grid(q1, 'C4')
        self.place_at_grid(q2, 'D5')
        self.place_at_grid(q3, 'E4')
        
        self.play(Write(q1), Write(q2), Write(q3))
        self.wait(2)
