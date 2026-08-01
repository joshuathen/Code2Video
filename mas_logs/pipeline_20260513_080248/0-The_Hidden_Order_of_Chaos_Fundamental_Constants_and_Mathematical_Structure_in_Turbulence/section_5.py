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
        self.setup_layout("Universal Constants and Kolmogorov Scales", [
            'The Kolmogorov length defines the smallest possible turbulent eddy.', 
            'Universal constants govern flow regardless of the physical system.', 
            'From Jupiter’s storms to a tea cup, math remains identical.', 
            'Zooming in reveals the same physics at microscopic scales.', 
            'Fundamental scales provide a bridge between chaos and order.'
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        # Define a small eddy (swirl)
        small_eddy = ParametricFunction(
            lambda t: np.array([0.2*t*np.cos(10*t), 0.2*t*np.sin(10*t), 0]),
            t_range=[0, 1],
            color="#00FFFF"
        )
        # Resolved Issue 35: Move to B4 and scale to 1.0
        self.place_at_grid(small_eddy, "B4", scale_factor=1.0)
        self.play(Create(small_eddy))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFA500")
        # Resolved Issue 27: Use SVG assets for storm cloud and tea cup
        storm_cloud = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/storm.svg").set_color("#FFA500")
        tea_cup = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cup.svg").set_color("#FFA500")
        
        self.place_at_grid(storm_cloud, "B2", scale_factor=0.7)
        self.place_at_grid(tea_cup, "B5", scale_factor=0.7)
        
        self.play(
            FadeIn(storm_cloud),
            FadeIn(tea_cup),
            FadeOut(small_eddy)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFC0CB")
        # Show similarity highlight (circles zooming in)
        highlight_1 = Circle(radius=0.5, color="#FFC0CB").move_to(storm_cloud)
        highlight_2 = Circle(radius=0.5, color="#FFC0CB").move_to(tea_cup)
        
        # Swirling fractal patterns revealed inside
        fractal_1 = ParametricFunction(
            lambda t: np.array([0.1*t*np.cos(12*t), 0.1*t*np.sin(12*t), 0]),
            t_range=[0, 2], color=WHITE
        ).move_to(storm_cloud)
        fractal_2 = fractal_1.copy().move_to(tea_cup)

        self.play(
            Create(highlight_1), 
            Create(highlight_2),
            FadeIn(fractal_1),
            FadeIn(fractal_2)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        # Formula for Kolmogorov length: η = (ν³ / ε)¼ 
        # Using VGroup of Text for manual layout without Tex
        formula_parts = VGroup(
            Text("η", color="#00FF00"),
            Text(" = (ν", color="#00FF00"),
            Text("³", font_size=16, color="#00FF00").shift(UP*0.15),
            Text(" / ε)", color="#00FF00"),
            Text("¼", font_size=20, color="#00FF00").shift(UP*0.2)
        ).arrange(RIGHT, buff=0.05)
        
        # Resolved Issue 37: Scale set to 1.0 to avoid crowding
        self.place_in_area(formula_parts, "D2", "D5", scale_factor=1.0)
        
        self.play(
            FadeIn(formula_parts),
            storm_cloud.animate.set_opacity(0.2),
            tea_cup.animate.set_opacity(0.2),
            highlight_1.animate.set_opacity(0.2),
            highlight_2.animate.set_opacity(0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFD700")
        # Kolmogorov Constant and bottom summary text
        ck_text = Text("C_K ≈ 1.5", color="#FFD700")
        ck_box = SurroundingRectangle(ck_text, color="#FFD700", buff=0.2)
        ck_group = VGroup(ck_text, ck_box)
        
        universal_text = Text("Universal at the Smallest Scales", color=WHITE, weight=BOLD)
        
        self.place_in_area(ck_group, "E3", "E4", scale_factor=1.0)
        # Resolved Issue 36: Positioned at F2-F5 with scale 0.6 to prevent cutoff
        self.place_in_area(universal_text, "F2", "F5", scale_factor=0.6)
        
        self.play(
            FadeIn(ck_group),
            Write(universal_text)
        )
        self.wait(2)
