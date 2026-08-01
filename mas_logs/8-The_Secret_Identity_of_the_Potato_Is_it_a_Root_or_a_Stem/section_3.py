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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout(
            "Defining the Tuber: A Specialized Storage Tank",
            [
                "A tuber is a modified plant part storing energy.",
                "Starch travels down the phloem to underground storage.",
                "Tubers can originate from either stems or roots."
            ]
        )

        # Color constants
        COLOR_TUBER = "#DEB887"
        COLOR_PHLOEM = "#4169E1"
        COLOR_STARCH = "#FFFFFF"
        COLOR_PLANT = "#228B22"
        COLOR_SOIL = "#8B4513"
        COLOR_LABEL = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_TUBER))
        
        # Soil line at row C
        soil_line = Line(self.grid["C1"], self.grid["C6"], color=COLOR_SOIL)
        
        # Plant above ground (Stem and Leaves)
        stem = Line(self.grid["C3"], self.grid["A3"], color=COLOR_PLANT, stroke_width=8)
        leaf_l = Triangle(color=COLOR_PLANT, fill_opacity=0.8).scale(0.3).rotate(PI/4)
        self.place_at_grid(leaf_l, "B2")
        leaf_r = Triangle(color=COLOR_PLANT, fill_opacity=0.8).scale(0.3).rotate(-PI/4)
        self.place_at_grid(leaf_r, "B4")
        
        # Beige storage bulb (tuber)
        tuber = Ellipse(width=1.8, height=1.2, color=COLOR_TUBER, fill_opacity=1.0)
        self.place_in_area(tuber, "D3", "E4")
        
        self.play(Create(soil_line), Create(stem), Create(leaf_l), Create(leaf_r))
        self.play(FadeIn(tuber, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_PHLOEM))
        
        # Blue phloem tube down the center
        phloem_tube = Line(self.grid["A3"], self.grid["D3"], color=COLOR_PHLOEM, stroke_width=4)
        
        # White starch particles
        p1 = Dot(radius=0.08, color=COLOR_STARCH)
        p2 = Dot(radius=0.08, color=COLOR_STARCH)
        p3 = Dot(radius=0.08, color=COLOR_STARCH)
        self.place_at_grid(p1, "A3")
        self.place_at_grid(p2, "A3")
        self.place_at_grid(p3, "A3")
        
        self.play(Create(phloem_tube))
        
        # Particles flow into the tuber
        self.play(p1.animate.move_to(self.grid["D3"]), run_time=1)
        self.play(p2.animate.move_to(self.grid["D3"]), run_time=1)
        self.play(
            p3.animate.move_to(self.grid["D3"]),
            tuber.animate.scale(1.2),
            run_time=1
        )
        self.play(FadeOut(p1, p2, p3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_LABEL))
        
        # Labels and Arrows
        label_stem = Text("Stem Tuber", font_size=20, color=COLOR_LABEL)
        self.place_at_grid(label_stem, "D1")
        arrow_stem = Arrow(self.grid["D1"], self.grid["D3"], color=COLOR_LABEL, buff=0.1)
        
        label_root = Text("Root Tuber", font_size=20, color=COLOR_LABEL)
        self.place_at_grid(label_root, "E6")
        arrow_root = Arrow(self.grid["E6"], self.grid["E4"], color=COLOR_LABEL, buff=0.1)
        
        self.play(
            Write(label_stem),
            GrowArrow(arrow_stem)
        )
        self.play(
            Write(label_root),
            GrowArrow(arrow_root)
        )
        self.wait(3)
