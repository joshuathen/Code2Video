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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Synthesis", [
            "Hyperspheres are essential extensions of 3D geometry.",
            "They are critical for understanding modern big data.",
            "Their volume distribution challenges our spatial intuition."
        ])

        # === Animation for Lecture Line 1 ===
        # Flash core concepts: N-dimensions, Volume, Concentration
        txt1 = Text("N-DIMENSIONS", color="#FFD700").scale(0.6)
        txt2 = Text("VOLUME", color="#FFD700").scale(0.6)
        txt3 = Text("CONCENTRATION", color="#FFD700").scale(0.6)
        
        concepts = VGroup(txt1, txt2, txt3).arrange(DOWN, buff=0.5)
        # Apply fix for Issue 33/39: Reposition concepts to A2-C5 to balance right grid
        self.place_in_area(concepts, "A2", "C5", scale_factor=0.75)
        
        self.play(FadeIn(concepts))
        self.lecture[0].set_color("#FFD700")
        self.play(Flash(concepts, color="#FFD700", line_length=0.2))

        # === Animation for Lecture Line 2 ===
        # Show summary table of key findings
        summary = Table(
            [["Dimension", "Geometry"], ["2D", "Circle"], ["3D", "Sphere"], ["ND", "Hypersphere"]],
            col_labels=[Text("System"), Text("Object")],
        ).scale(0.4)
        
        # Apply fix for Issue 32/34/39: Reposition table to D2-F6 to avoid clutter/improve layout
        self.place_in_area(summary, "D2", "F6", scale_factor=0.80)
        self.play(FadeOut(concepts), FadeIn(summary))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        # Fade out all elements
        self.lecture[2].set_color("#FFFFFF")
        self.play(FadeOut(summary), FadeOut(self.lecture), FadeOut(self.title))
        self.wait(1)
