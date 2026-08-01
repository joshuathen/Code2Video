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
        # Setup data
        title_text = "Introduction: The Impossible Connection"
        lecture_lines = [
            "Imagine a small block between a wall and large block.",
            "The large block slides toward the small one.",
            "How many collisions will occur in this setup?"
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Wall: centered at column 1, spans rows A to F
        wall = Rectangle(width=0.2, height=5.0, color="#FFFFFF", fill_opacity=1)
        self.place_in_area(wall, "A1", "F1")
        
        # Floor: spanning column 1 to 6
        floor = Line(LEFT, RIGHT, color=WHITE).scale(2.5)
        self.place_in_area(floor, "F1", "F6")
        
        # Small block: color #52C41A
        small_block = Square(side_length=0.6, color="#52C41A", fill_opacity=0.8)
        self.place_at_grid(small_block, "E2")
        
        # Large block: color #1890FF
        large_block = Square(side_length=1.2, color="#1890FF", fill_opacity=0.8)
        self.place_at_grid(large_block, "E5")
        
        # Labels
        m_label = MathTex("m", color="#52C41A")
        M_label = MathTex("M", color="#1890FF")
        self.place_at_grid(m_label, "D2", scale_factor=0.7)
        self.place_at_grid(M_label, "D5", scale_factor=0.7)

        self.play(
            Create(wall),
            Create(floor),
            FadeIn(small_block),
            FadeIn(large_block),
            Write(m_label),
            Write(M_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(WHITE)
        
        # Sliding motion using ValueTracker
        slide_tracker = ValueTracker(0)
        start_pt = self.grid["E5"]
        end_pt = self.grid["E3"]
        
        # Update large block and its label
        large_block.add_updater(lambda m: m.move_to(interpolate(start_pt, end_pt, slide_tracker.get_value())))
        M_label.add_updater(lambda m: m.next_to(large_block, UP, buff=0.2))
        
        self.play(slide_tracker.animate.set_value(1), run_time=2, rate_func=linear)
        self.wait(1)
        
        large_block.clear_updaters()
        M_label.clear_updaters()

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(WHITE)
        
        question_text = Text("How many collisions will occur?", font_size=24, color=WHITE)
        self.place_in_area(question_text, 'B3', 'C6', scale_factor=0.5)
        
        self.play(Write(question_text))
        self.wait(3)
