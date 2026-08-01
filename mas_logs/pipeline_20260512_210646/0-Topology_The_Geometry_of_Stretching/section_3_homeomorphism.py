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

class Section3HomeomorphismScene(TeachingScene):
    def construct(self):
        title = "The Golden Rule: Homeomorphism"
        lines = [
            "Homeomorphism defines valid transformations in topology.",
            "You can stretch, bend, and squish any shape.",
            "However, you cannot tear or puncture the material.",
            "Gluing separate points together is also strictly forbidden.",
            "These rules preserve the object's fundamental structure."
        ]
        self.setup_layout(title, lines)

        # Colors
        ORANGE_BLOB = "#FFA500"
        WHITE_LABEL = "#FFFFFF"
        RED_X = "#FF0000"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create an orange blob shape (using a rounded rectangle for 'dough' feel)
        blob = RoundedRectangle(corner_radius=1, height=2, width=2, fill_color=ORANGE_BLOB, fill_opacity=1, stroke_width=0)
        # Resolved Issue 42: blob area adjusted to B2-D5 to prevent overlap with labels
        self.place_in_area(blob, "B2", "D5", scale_factor=1.0)
        
        label = Text("Play-dough", font_size=20, color=WHITE_LABEL)
        # Resolved Issue 41: label moved to area E2-E4 and scaled down to avoid cut-off
        self.place_in_area(label, 'E2', 'E4', scale_factor=0.8)

        self.play(FadeIn(blob), Write(label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Stretch horizontal
        self.play(blob.animate.stretch(2.5, 0), run_time=1)
        self.play(blob.animate.stretch(1/2.5, 0), run_time=1)
        
        # Squish vertical
        self.play(blob.animate.stretch(0.3, 1), run_time=1)
        self.play(blob.animate.stretch(1/0.3, 1), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        # Puncture hole
        # Centers hole on the current blob position
        hole = Circle(radius=0.4, color=BLACK, fill_opacity=1).move_to(blob.get_center())
        cross_mark = Cross(blob, stroke_color=RED_X, stroke_width=10)
        
        self.play(FadeIn(hole))
        self.play(Create(cross_mark))
        self.wait(1)
        
        self.play(FadeOut(cross_mark), FadeOut(hole))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)

        # Cut into two pieces
        blob_left = RoundedRectangle(corner_radius=0.5, height=2, width=0.9, fill_color=ORANGE_BLOB, fill_opacity=1, stroke_width=0)
        blob_right = RoundedRectangle(corner_radius=0.5, height=2, width=0.9, fill_color=ORANGE_BLOB, fill_opacity=1, stroke_width=0)
        
        blob_left.move_to(blob.get_center() + LEFT*0.55)
        blob_right.move_to(blob.get_center() + RIGHT*0.55)
        
        self.remove(blob)
        self.add(blob_left, blob_right)
        
        cross_mark_2 = Cross(VGroup(blob_left, blob_right), stroke_color=RED_X, stroke_width=10)
        
        self.play(blob_left.animate.shift(LEFT*0.5), blob_right.animate.shift(RIGHT*0.5))
        self.play(Create(cross_mark_2))
        self.wait(1)
        
        self.play(FadeOut(cross_mark_2), FadeOut(blob_left), FadeOut(blob_right))
        self.add(blob)
        self.play(FadeIn(blob))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)

        # Pulse to show structure preserved
        self.play(blob.animate.scale(1.1), run_time=0.5)
        self.play(blob.animate.scale(1/1.1), run_time=0.5)
        self.wait(2)
