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
        # Setup the scene with title and lecture lines
        self.setup_layout(
            "Application: The Delivery Drone", 
            [
                "A drone arrives within a ten-minute window.", 
                "We model this with a uniform rectangular PDF.", 
                "To find the chance it arrives within two minutes...", 
                "...calculate the area of that two-minute slice.", 
                "The area of the slice is the event's probability."
            ]
        )
        
        # Define colors
        HIGHLIGHT_COLOR = YELLOW
        PDF_COLOR = BLUE
        SHADE_COLOR = "#00FA9A" # Medium Spring Green
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # X-axis line (0 to 10 minutes) starting at Col 2
        x_axis = Line(self.grid["E2"], self.grid["E6"], color=WHITE)
        tick_0 = Line(self.grid["E2"] + DOWN*0.1, self.grid["E2"] + UP*0.1, color=WHITE)
        tick_10 = Line(self.grid["E6"] + DOWN*0.1, self.grid["E6"] + UP*0.1, color=WHITE)
        
        label_0 = Text("0", font_size=18, color=WHITE)
        self.place_at_grid(label_0, "F2") # Issue 53 Fix
        
        # Drone asset - Issue 39
        drone = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/drone.svg")
        self.place_at_grid(drone, "B2", scale_factor=0.4)
        
        # PDF Rectangle: Covers width from Col 2 to Col 6 (4 units) and height of 2 units (PDF height 0.1)
        # Issue 52 Fix
        pdf_rect = Rectangle(width=4.0, height=2.0, stroke_color=PDF_COLOR, stroke_width=3)
        self.place_in_area(pdf_rect, "C2", "E6")
        
        self.play(Create(x_axis), Create(tick_0), Create(tick_10), FadeIn(label_0))
        self.play(Create(pdf_rect), FadeIn(drone))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        y_axis = Line(self.grid["E2"], self.grid["C2"], color=WHITE)
        h_label = Text("0.1", font_size=20, color=WHITE)
        self.place_at_grid(h_label, "C1")
        
        label_10_min = Text("10 minutes", font_size=18, color=WHITE)
        self.place_at_grid(label_10_min, "F6")
        
        self.play(Create(y_axis), FadeIn(h_label), FadeIn(label_10_min))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Shade slice from 0 to 2 minutes. 
        # Total width 10 min = 4 units. 2 min = 0.8 units.
        shade = Rectangle(
            width=0.8, 
            height=2.0, 
            fill_color=SHADE_COLOR, 
            fill_opacity=0.6, 
            stroke_width=0
        )
        # Center of PDF rectangle is (3.5, -0.8). Rect starts at x=1.5. 
        # Shade covers x=[1.5, 2.3]. Center of shade is at 1.9.
        shade.move_to([1.9, -0.8, 0])
        
        interval_label = Text("0 to 2 min", font_size=16, color=SHADE_COLOR)
        self.place_at_grid(interval_label, "B2") # Issue 54 Fix
        
        self.play(FadeIn(shade), FadeIn(interval_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        math_text = Text("2 min * 0.1 = 0.2", font_size=22, color=WHITE)
        self.place_at_grid(math_text, "B4")
        
        self.play(Write(math_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        prob_label = Text("20% Probability", font_size=24, color=WHITE)
        self.place_at_grid(prob_label, "A4")
        
        self.play(FadeIn(prob_label))
        self.wait(2)
        
        # Final cleanup for the lecture line color
        self.lecture[4].set_color(WHITE)
        self.wait(1)
