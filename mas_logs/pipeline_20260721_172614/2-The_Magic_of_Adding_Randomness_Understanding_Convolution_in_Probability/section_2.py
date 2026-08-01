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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "Prerequisite Check: Independence and PDFs"
        lines = [
            "First, our random variables must be independent of each other.",
            "Each variable has a Probability Density Function, or PDF.",
            "This PDF represents the 'shape' of its uncertainty."
        ]
        
        self.setup_layout(title_text, lines)

        # Colors from storyboard
        COLOR_PDF1 = "#00FFFF"
        COLOR_PDF2 = "#FF00FF"
        COLOR_SUM = "#FFFF00"
        COLOR_TEXT = "#FFFFFF"
        HIGHLIGHT = "#FFFF00"

        # Asset path
        BATTERY_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/battery.svg"

        # === Animation for Lecture Line 1 ===
        # "First, our random variables must be independent of each other."
        self.lecture[0].set_color(HIGHLIGHT)
        
        # Load and color battery icons (representing PDF shapes)
        # SVGMobjects are loaded once as per performance constraints
        battery1 = SVGMobject(BATTERY_ASSET).set_color(COLOR_PDF1)
        battery2 = SVGMobject(BATTERY_ASSET).set_color(COLOR_PDF2)
        
        # Position at C2 and C5 to avoid being too high (Issue 40)
        self.place_at_grid(battery1, "C2", scale_factor=0.8)
        self.place_at_grid(battery2, "C5", scale_factor=0.8)
        
        self.play(DrawBorderThenFill(battery1), DrawBorderThenFill(battery2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Each variable has a Probability Density Function, or PDF."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT)
        
        label1 = Text("PDF 1", font_size=24, color=COLOR_TEXT)
        label2 = Text("PDF 2", font_size=24, color=COLOR_TEXT)
        
        # Position labels in area with appropriate scaling (Issue 40)
        self.place_in_area(label1, "B2", "B3", scale_factor=0.8)
        self.place_in_area(label2, "B5", "B6", scale_factor=0.8)
        
        self.play(Write(label1), Write(label2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This PDF represents the 'shape' of its uncertainty."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT)
        
        # Central sum node positioned at F4 (Issue 40)
        sum_circle = Circle(radius=0.5, color=COLOR_SUM)
        sum_text = Text("Sum", font_size=24, color=COLOR_SUM)
        sum_group = VGroup(sum_circle, sum_text)
        
        self.place_at_grid(sum_group, "F4", scale_factor=0.8)
        
        # Arrows from batteries to the central sum node
        arrow1 = Arrow(
            start=battery1.get_bottom(), 
            end=sum_group.get_top(), 
            color=COLOR_TEXT, 
            buff=0.1
        )
        arrow2 = Arrow(
            start=battery2.get_bottom(), 
            end=sum_group.get_top(), 
            color=COLOR_TEXT, 
            buff=0.1
        )
        
        self.play(FadeIn(sum_group))
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))
        self.wait(2)
        
        # Reset lecture line color
        self.lecture[2].set_color(WHITE)
        self.wait(1)
