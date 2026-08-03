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
        # Initial Setup
        title_text = "Summary: The Universal Blueprint"
        lecture_lines = [
            "Turbulence constants reveal a hidden, universal structure.",
            "From coffee cups to galaxies, the math holds.",
            "Order exists within the heart of chaos."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a small soup bowl icon (#D2B48C) on left and a large galaxy spiral (#4B0082) on right.
        
        # Soup Bowl Construction
        bowl_bottom = Arc(start_angle=PI, angle=PI, radius=0.6, color="#D2B48C")
        bowl_top = Line(bowl_bottom.get_start(), bowl_bottom.get_end(), color="#D2B48C")
        # Steam effect
        steam1 = ParametricFunction(lambda t: np.array([0.1*np.sin(4*t), t, 0]), t_range=[0, 0.4], color="#D2B48C").shift(UP*0.1 + LEFT*0.2)
        steam2 = ParametricFunction(lambda t: np.array([0.1*np.sin(4*t), t, 0]), t_range=[0, 0.4], color="#D2B48C").shift(UP*0.1 + RIGHT*0.2)
        soup_bowl = VGroup(bowl_bottom, bowl_top, steam1, steam2)
        
        # Galaxy Spiral Construction
        galaxy_spiral = ParametricFunction(
            lambda t: np.array([0.1 * t * np.cos(t), 0.1 * t * np.sin(t), 0]),
            t_range=[0, 5 * PI],
            color="#4B0082"
        )
        galaxy_core = Dot(radius=0.1, color="#4B0082")
        galaxy_group = VGroup(galaxy_spiral, galaxy_core)

        # Positioning soup bowl (Left-ish of the animation area)
        self.place_at_grid(soup_bowl, "C2", scale_factor=0.8)
        # Positioning galaxy (Right-ish of the animation area)
        # Fixed scale factor from 1.2 to 1.0 (Issue 40)
        self.place_at_grid(galaxy_group, "C5", scale_factor=1.0)

        # Highlight first lecture line
        self.lecture[0].set_color(YELLOW)
        self.play(
            FadeIn(soup_bowl, shift=RIGHT),
            FadeIn(galaxy_group, shift=LEFT),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Overlay yellow (#FFFF00) -5/3 slope line on both.
        # Draw a green (#00FF00) bridge connecting both visuals.
        
        # Define line with slope -5/3 (dx=0.6, dy=-1.0)
        slope_line_base = Line(
            start=UP*0.5 + LEFT*0.3,
            end=DOWN*0.5 + RIGHT*0.3,
            color="#FFFF00",
            stroke_width=6
        )
        
        slope_line_bowl = slope_line_base.copy()
        slope_line_galaxy = slope_line_base.copy()
        
        # Fixed scale factor from 1.0 to 0.8 for soup bowl slope (Issue 41)
        self.place_at_grid(slope_line_bowl, "C2", scale_factor=0.8)
        self.place_at_grid(slope_line_galaxy, "C5", scale_factor=1.0)
        
        # Bridge connecting both centers
        bridge = DoubleArrow(
            self.grid["C2"] + RIGHT * 0.7,
            self.grid["C5"] + LEFT * 0.7,
            color="#00FF00",
            buff=0,
            stroke_width=5
        )

        # Update lecture highlighting
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(
            Create(slope_line_bowl),
            Create(slope_line_galaxy),
            run_time=1.5
        )
        self.play(Create(bridge), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final text 'Universal Math' appears in gold (#FFD700) center screen.
        
        universal_math = Text("Universal Math", font_size=40, color="#FFD700", weight=BOLD)
        # Shifted from Row E to Row D for better visual flow (Issue 39)
        self.place_in_area(universal_math, "D2", "D5", scale_factor=0.9)

        # Update lecture highlighting
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(Write(universal_math), run_time=2)
        
        # Final emphasis pulse
        self.play(
            universal_math.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(3)
