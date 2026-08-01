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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Big Question: Why Sum Variables?", [
            "We often combine independent random processes.",
            "Let Z be the sum of X and Y.",
            "Total delivery time equals travel plus unloading time."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display formula Z = X + Y in #FFFFFF. 
        # Refined scale (1.2) and position (B2-B5) per Issue 46.
        formula = MathTex("Z = X + Y", color=WHITE)
        self.place_in_area(formula, "B2", "B5", scale_factor=1.2)
        
        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            Write(formula)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Integrate SVG assets and refine scale per Issue 46.
        
        # Character 'Delivery Robot' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot.set_color("#ADD8E6")
        self.place_at_grid(robot, "D2", scale_factor=0.9)
        
        # 'Travel Time' (X) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/travel.svg]
        asset_x = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/travel.svg")
        asset_x.set_color("#FFFF00")
        label_x = MathTex("X", color="#FFFF00").scale(0.8)
        x_comp = VGroup(asset_x, label_x).arrange(UP, buff=0.1)
        self.place_at_grid(x_comp, "D3", scale_factor=0.9)
        
        # 'Unloading Time' (Y) (Keep manual representation as no specific asset provided)
        bar_y = Rectangle(width=0.8, height=0.5, color="#00FF00", fill_opacity=0.8)
        label_y = MathTex("Y", color="#00FF00").scale(0.8)
        y_comp = VGroup(bar_y, label_y).arrange(UP, buff=0.1)
        self.place_at_grid(y_comp, "D4", scale_factor=0.9)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            FadeIn(robot),
            FadeIn(x_comp),
            FadeIn(y_comp)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight Z as the total span.
        # Moved to E2-E5 with scale 1.1 to avoid edge crowding per Issue 46.
        
        # Reuse visuals for consistency
        x_vis = asset_x.copy()
        y_vis = bar_y.copy()
        combined_visuals = VGroup(x_vis, y_vis).arrange(RIGHT, buff=0.2)
        
        combined_bars_group = VGroup(combined_visuals)
        self.place_in_area(combined_bars_group, "E2", "E5", scale_factor=1.1)
        
        bracket = Brace(combined_visuals, direction=DOWN, color="#FF69B4")
        z_label = MathTex("Z = X + Y", color="#FF69B4").scale(0.8).next_to(bracket, DOWN, buff=0.1)
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            TransformFromCopy(VGroup(asset_x, bar_y), combined_visuals),
            Create(bracket),
            FadeIn(z_label)
        )
        self.wait(3)
