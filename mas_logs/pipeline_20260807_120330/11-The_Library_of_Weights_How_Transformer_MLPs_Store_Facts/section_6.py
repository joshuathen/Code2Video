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
        # Colors
        COLOR_GEO = "#FF00FF"
        COLOR_FACT = "#FFFFFF"
        COLOR_PULSE = "#ADD8E6"
        COLOR_HIGHLIGHT = "#FFFF00"
        
        self.setup_layout(
            "Superposition: A Multi-Dimensional Library", 
            [
                "High-dimensional space allows for massive storage capacity.",
                "Millions of facts coexist without interfering with others.",
                "This phenomenon is known as neural superposition."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # High-dimensional space allows for massive storage capacity.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg]
        library_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        self.place_at_grid(library_icon, 'A1', scale_factor=0.6)
        library_icon.set_color(COLOR_FACT)

        # Create a dense cloud of points
        np.random.seed(42)
        num_points = 80
        # Create points initially centered at origin to allow place_in_area to work predictably
        points = VGroup(*[
            Dot(radius=0.04, color=COLOR_FACT).move_to([
                np.random.uniform(-2, 2), 
                np.random.uniform(-2, 2), 
                0
            ]) for _ in range(num_points)
        ])
        
        # [Fix 43]: place_in_area with scale_factor=0.6 to avoid overlapping lecture text
        self.place_in_area(points, 'A1', 'F6', scale_factor=0.6)

        self.lecture[0].set_color(COLOR_FACT)
        self.play(FadeIn(library_icon), FadeIn(points))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Millions of facts coexist without interfering with others.
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg]
        globe_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        self.place_at_grid(globe_icon, 'D1', scale_factor=0.6)
        globe_icon.set_color(COLOR_GEO)

        # Color some points to represent "Geography" topic
        geography_points = points[:20]
        science_points = points[20:40]
        
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(COLOR_FACT),
            geography_points.animate.set_color(COLOR_GEO),
            science_points.animate.set_color("#00FFFF"), # Cyan for Science
            FadeIn(globe_icon)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This phenomenon is known as neural superposition.
        
        # [Fix 44]: arrow1 at B2, fact1 at B4
        arrow1 = Arrow(start=LEFT, end=RIGHT, color=COLOR_HIGHLIGHT, buff=0).scale(0.5)
        self.place_at_grid(arrow1, 'B2')
        fact1 = Dot(radius=0.1, color=COLOR_HIGHLIGHT)
        self.place_at_grid(fact1, 'B4')
        
        # [Fix 45]: arrow2 at E2, fact2 at E4
        arrow2 = Arrow(start=LEFT, end=RIGHT, color=COLOR_HIGHLIGHT, buff=0).scale(0.5)
        self.place_at_grid(arrow2, 'E2')
        fact2 = Dot(radius=0.1, color=COLOR_HIGHLIGHT)
        self.place_at_grid(fact2, 'E4')

        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(COLOR_PULSE),
            Create(arrow1), Create(fact1),
            Create(arrow2), Create(fact2)
        )
        
        # Pulse the cloud to symbolize superposition
        self.play(
            points.animate.set_color(COLOR_PULSE).scale(1.1),
            rate_func=there_and_back,
            run_time=2
        )
        
        self.wait(2)
