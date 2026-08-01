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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initializing the layout with the lecture lines and title
        title_text = "Summary & Mastery Recap"
        lecture_lines = [
            "First, differentiate every term, using the chain rule.",
            "Next, use algebra to solve for dy dx.",
            "Now you can find slopes for any tangled curve."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define specific colors for synchronization
        COL_STEP1 = "#FFFF00"  # Yellow
        COL_STEP2 = "#ADD8E6"  # Light Blue
        COL_STEP3 = "#FFA500"  # Orange
        COL_GOLD = "#FFD700"   # Gold

        # === Animation for Lecture Line 1 ===
        # Show a numbered list: 1. Differentiate, 2. Isolate dy/dx, 3. Solve for slope.
        step1 = Text("1. Differentiate", font_size=32, color=COL_STEP1)
        step2 = Text("2. Isolate dy/dx", font_size=32, color=COL_STEP2)
        step3 = Text("3. Solve for slope", font_size=32, color=COL_STEP3)

        # Positioning steps on the right side grid (A-C rows)
        self.place_at_grid(step1, "A3", scale_factor=0.8)
        self.place_at_grid(step2, "B3", scale_factor=0.8)
        self.place_at_grid(step3, "C3", scale_factor=0.8)

        self.play(
            self.lecture[0].animate.set_color(COL_STEP1),
            Write(step1),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight step 2 and change lecture line color
        self.play(
            self.lecture[1].animate.set_color(COL_STEP2),
            Write(step2),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight step 3, change lecture line color, and show the figure-8 curve
        self.play(
            self.lecture[2].animate.set_color(COL_STEP3),
            Write(step3),
            run_time=1.5
        )
        self.wait(0.5)

        # Complex figure-8 (Lissajous) curve: x = sin(2t), y = sin(t)
        # Positioned in the D1-F6 area
        lissajous = ParametricFunction(
            lambda t: np.array([1.5 * np.sin(2 * t), 0.8 * np.sin(t), 0]),
            t_range=[0, TAU],
            color=COL_STEP3,
            stroke_width=4
        )
        self.place_in_area(lissajous, "D1", "F6", scale_factor=1.0)

        # Function to generate tangent arrows based on curve proportion
        def get_tangent_arrow(proportion, color):
            p = lissajous.point_from_proportion(proportion)
            # Find a point slightly further to determine direction
            p_next = lissajous.point_from_proportion((proportion + 0.01) % 1.0)
            direction = (p_next - p) / np.linalg.norm(p_next - p) * 0.7
            return Arrow(p, p + direction, buff=0, color=color, stroke_width=4)

        # Arrows at various points of the curve
        arrows = VGroup(
            get_tangent_arrow(0.0, WHITE),    # Crossing center
            get_tangent_arrow(0.125, COL_STEP1), # Top right lobe
            get_tangent_arrow(0.5, WHITE),    # Crossing center back
            get_tangent_arrow(0.625, COL_STEP2)  # Bottom left lobe
        )

        self.play(Create(lissajous), run_time=2)
        self.play(Create(arrows), run_time=1.5)
        self.wait(2)

        # === Concluding Mastery Highlight ===
        # Text 'Implicit Differentiation: Unlocking Hidden Curves' in gold
        concluding_text = Text(
            "Implicit Differentiation:\nUnlocking Hidden Curves",
            font_size=38,
            color=COL_GOLD,
            weight=BOLD,
            line_spacing=0.8
        )
        
        # Clear specific animation elements to focus on the final message
        self.play(
            FadeOut(step1, step2, step3, lissajous, arrows),
            run_time=1
        )
        
        # Place in central area of the right grid
        self.place_in_area(concluding_text, "A1", "F6", scale_factor=1.0)
        self.play(Write(concluding_text), run_time=2)
        self.wait(3)
