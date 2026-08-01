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
        # Define lecture lines for the scene
        lecture_lines = [
            'Light bends whenever its speed changes between mediums.', 
            'Denser materials pull the light toward the normal line.', 
            'Next, we explore the limit where light reflects entirely.'
        ]
        self.setup_layout("Summary and Synthesis", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line (Light Blue for Water/Speed context)
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        
        # Setup media layers: Air, Water, Glass, Air
        # Water layer (Rows B and C)
        water_rect = Rectangle(width=6.0, height=2.0, fill_opacity=0.3, fill_color="#ADD8E6", stroke_width=0)
        self.place_in_area(water_rect, "B1", "C6")
        
        # Glass layer (Rows D and E)
        glass_rect = Rectangle(width=6.0, height=2.0, fill_opacity=0.5, fill_color="#4682B4", stroke_width=0)
        self.place_in_area(glass_rect, "D1", "E6")

        # Media Labels
        air_label_1 = Text("Air (n=1.0)", font_size=16, color=WHITE)
        self.place_at_grid(air_label_1, "A1")
        
        # water_label placement (Issue 45)
        water_label = Text("Water (n=1.33)", font_size=16, color="#ADD8E6")
        self.place_in_area(water_label, "B1", "B2", scale_factor=0.8)
        
        # glass_label placement (Issue 46)
        glass_label = Text("Glass (n=1.5)", font_size=16, color="#4682B4")
        self.place_in_area(glass_label, "D1", "D2", scale_factor=0.8)
        
        air_label_2 = Text("Air (n=1.0)", font_size=16, color=WHITE)
        self.place_at_grid(air_label_2, "F1")

        # Load Assets (Issue 28)
        water_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/water.svg")
        self.place_in_area(water_icon, "C5", "C6", scale_factor=0.5)
        
        glass_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/glass.svg")
        self.place_in_area(glass_icon, "E5", "E6", scale_factor=0.5)

        self.play(
            FadeIn(water_rect), 
            FadeIn(glass_rect), 
            FadeIn(air_label_1), 
            FadeIn(water_label), 
            FadeIn(glass_label), 
            FadeIn(air_label_2),
            FadeIn(water_icon),
            FadeIn(glass_icon)
        )

        # Initial Ray segment in Air
        # Start at A2
        p0 = self.grid["A2"]
        # Intersection with Water boundary (y=1.7)
        p1 = np.array([2.0, 1.7, 0])
        ray1 = Line(p0, p1, color=YELLOW, stroke_width=4)
        
        self.play(Create(ray1))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Define bending points based on Snell's Law
        # Boundary Water/Glass (y=-0.3)
        p2 = np.array([3.25, -0.3, 0])
        # Boundary Glass/Air (y=-2.3)
        p3 = np.array([4.31, -2.3, 0])
        # Final exit path point (y=-2.8)
        p4 = np.array([4.81, -2.8, 0])

        # Normals for each interface
        n1 = DashedLine(p1 + UP*0.6, p1 + DOWN*0.6, color=GREY_A)
        n2 = DashedLine(p2 + UP*0.6, p2 + DOWN*0.6, color=GREY_A)
        n3 = DashedLine(p3 + UP*0.6, p3 + DOWN*0.6, color=GREY_A)
        
        self.play(Create(n1), Create(n2), Create(n3))

        # Sequentially animate ray bending
        ray2 = Line(p1, p2, color=YELLOW, stroke_width=4)
        ray3 = Line(p2, p3, color=YELLOW, stroke_width=4)
        ray4 = Line(p3, p4, color=YELLOW, stroke_width=4)

        self.play(Create(ray2), run_time=1)
        self.play(Create(ray3), run_time=1)
        self.play(Create(ray4), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Return focus to line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Final summary text (Issue 44)
        summary_txt = Text("Refraction: Light bends when it changes speed.", font_size=20, color=WHITE)
        self.place_in_area(summary_txt, "A2", "A4", scale_factor=0.7)
        
        self.play(Write(summary_txt))
        self.wait(2)
