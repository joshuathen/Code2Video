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
        # Setup the basic layout
        lecture_lines = [
            "Though both lanes are infinite, they don't race equally.",
            "Primes of form 4n plus 3 usually take the lead.",
            "This phenomenon is known as the famous Chebyshev’s Bias.",
            "The lead fluctuates, but Team 3 stays ahead most often.",
            "A subtle asymmetry exists within the randomness of primes."
        ]
        self.setup_layout("The Prime Race: Chebyshev’s Bias", lecture_lines)

        MAGENTA_CLR = "#FF00FF"
        CYAN_CLR = "#00FFFF"
        YELLOW_CLR = "#FFFF00"

        # Define Grid Boundaries for the Graph
        # Origin at F2 (Bottom Left of the race area)
        origin = self.grid["F2"]
        x_end = self.grid["F6"]
        y_end = self.grid["B2"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create Axes
        x_axis = Arrow(start=origin, end=x_end + RIGHT*0.5, color=WHITE, buff=0)
        y_axis = Arrow(start=origin, end=y_end + UP*0.5, color=WHITE, buff=0)
        
        x_label = Text("Search Limit", font_size=16).next_to(x_axis, DOWN)
        y_label = Text("Prime Count", font_size=16).rotate(90*DEGREES).next_to(y_axis, LEFT)
        
        self.play(Create(x_axis), Create(y_axis), FadeIn(x_label), FadeIn(y_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Match color to Team 3 (4n+3)
        self.play(self.lecture[1].animate.set_color(MAGENTA_CLR))
        
        # Points for the race (conceptual data for 4n+3 and 4n+1)
        m_pts = [
            origin,
            origin + RIGHT*0.4 + UP*0.2, # 3
            origin + RIGHT*0.8 + UP*0.4, # 7
            origin + RIGHT*1.2 + UP*0.6, # 11
            origin + RIGHT*1.6 + UP*0.6, 
            origin + RIGHT*2.0 + UP*0.8, # 19
            origin + RIGHT*2.4 + UP*1.0, # 23
            origin + RIGHT*2.8 + UP*1.0, 
            origin + RIGHT*3.2 + UP*1.2, # 31
            origin + RIGHT*3.6 + UP*1.4, # 43
            origin + RIGHT*4.0 + UP*1.6, # 47
            origin + RIGHT*4.4 + UP*1.8,
        ]
        
        c_pts = [
            origin,
            origin + RIGHT*0.6 + UP*0.2, # 5
            origin + RIGHT*1.4 + UP*0.4, # 13
            origin + RIGHT*1.8 + UP*0.6, # 17
            origin + RIGHT*2.5 + UP*0.7, 
            origin + RIGHT*3.0 + UP*0.8, # 29
            origin + RIGHT*3.4 + UP*1.0, # 37
            origin + RIGHT*3.8 + UP*1.2, # 41
            origin + RIGHT*4.4 + UP*1.3,
        ]

        magenta_line = VMobject(color=MAGENTA_CLR)
        magenta_line.set_points_as_corners(m_pts)
        
        cyan_line = VMobject(color=CYAN_CLR)
        cyan_line.set_points_as_corners(c_pts)

        team3_label = Text("Team 3 (4n+3)", font_size=14, color=MAGENTA_CLR)
        self.place_at_grid(team3_label, "A2") # Fixed Issue 42
        
        team1_label = Text("Team 1 (4n+1)", font_size=14, color=CYAN_CLR)
        self.place_at_grid(team1_label, "A4") # Fixed Issue 42

        self.play(Create(magenta_line), Create(cyan_line), FadeIn(team3_label), FadeIn(team1_label), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW_CLR))
        
        bias_text = Text("Chebyshev's Bias", font_size=32, color=YELLOW_CLR)
        self.place_in_area(bias_text, "B2", "C4") # Fixed Issue 40
        
        self.play(Write(bias_text))
        self.play(Flash(bias_text, color=YELLOW_CLR, num_lines=12))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(MAGENTA_CLR))
        
        # Demonstrate the Lead
        lead_arrow = Arrow(
            start=origin + RIGHT*2.4 + UP*0.4, 
            end=origin + RIGHT*2.4 + UP*0.9, 
            color=WHITE, stroke_width=2, max_tip_length_to_length_ratio=0.2
        )
        lead_label = Text("The Lead", font_size=14).next_to(lead_arrow, DOWN, buff=0.1)
        
        self.play(GrowArrow(lead_arrow), FadeIn(lead_label))
        self.wait(0.5)
        
        # Simulate a crossing point (visual representation of fluctuation)
        crossing_cyan = VMobject(color=CYAN_CLR)
        cx_pts = [
            origin + RIGHT*4.4 + UP*1.3,
            origin + RIGHT*4.8 + UP*2.0, 
            origin + RIGHT*5.2 + UP*2.1
        ]
        mx_pts = [
            origin + RIGHT*4.4 + UP*1.8,
            origin + RIGHT*4.8 + UP*1.9, 
            origin + RIGHT*5.2 + UP*2.4  
        ]
        
        crossing_cyan.set_points_as_corners(cx_pts)
        crossing_magenta = VMobject(color=MAGENTA_CLR)
        crossing_magenta.set_points_as_corners(mx_pts)
        
        self.play(
            Create(crossing_cyan),
            Create(crossing_magenta),
            FadeOut(lead_arrow),
            FadeOut(lead_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(CYAN_CLR))
        
        # Highlight asymmetry (Yellow box around the bias area)
        highlight_box = Rectangle(width=4.0, height=1.0, color=YELLOW_CLR, stroke_width=1)
        self.place_in_area(highlight_box, "B1", "C5") # Fixed Issue 41
        
        self.play(Create(highlight_box))
        self.play(Indicate(highlight_box))
        self.wait(2)
