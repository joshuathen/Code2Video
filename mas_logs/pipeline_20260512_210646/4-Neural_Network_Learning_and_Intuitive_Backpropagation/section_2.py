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
        # Setup the layout with Title and Lecture Lines
        self.setup_layout(
            "Prerequisite: The Error Landscape (Loss Function)", 
            [
                "Imagine Nero's error as a hilly 3D terrain.", 
                "Peaks mean high error; valleys mean perfect accuracy.", 
                "We need to find the path downwards."
            ]
        )

        # Colors
        BLUE_CONTOUR = "#1E90FF"
        RED_PEAK = "#FF0000"
        GREEN_VALLEY = "#00FF00"
        HIGHLIGHT_YELLOW = "#FFFF00"
        INACTIVE_GREY = "#888888"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_YELLOW))
        
        # Create 2D landscape with blue contour lines
        # Peak Contours moved to C5 as per Issue 61
        peak_contours = VGroup(*[
            Ellipse(width=2.5-i*0.5, height=1.8-i*0.4, color=BLUE_CONTOUR, stroke_width=2)
            for i in range(4)
        ])
        self.place_at_grid(peak_contours, "C5")
        
        # Valley Contours at E2
        valley_contours = VGroup(*[
            Ellipse(width=2.2-i*0.4, height=1.5-i*0.3, color=BLUE_CONTOUR, stroke_width=2)
            for i in range(4)
        ])
        self.place_at_grid(valley_contours, "E2")
        
        # Larger background contours connecting them
        bg_contour = Ellipse(width=5, height=4, color=BLUE_CONTOUR, stroke_width=1, stroke_opacity=0.5)
        self.place_in_area(bg_contour, "A1", "F6")

        self.play(
            Create(peak_contours),
            Create(valley_contours),
            Create(bg_contour),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(INACTIVE_GREY),
            self.lecture[1].animate.set_color(HIGHLIGHT_YELLOW)
        )

        # Place a red dot and Nero icon on a high 'Error Peak' at C5
        error_dot = Dot(color=RED_PEAK, radius=0.12)
        self.place_at_grid(error_dot, "C5")
        
        # Asset: Nero Icon integration
        nero_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/nero.svg")
        self.place_at_grid(nero_icon, "C5", scale_factor=0.35)
        
        # Reposition peak_label to area B5-B6 to avoid top edge clipping
        peak_label = Text("Error Peak", font_size=20, color=RED_PEAK)
        self.place_in_area(peak_label, "B5", "B6", scale_factor=0.8)

        self.play(
            FadeIn(error_dot, scale=1.5),
            FadeIn(nero_icon),
            Write(peak_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(INACTIVE_GREY),
            self.lecture[2].animate.set_color(HIGHLIGHT_YELLOW)
        )

        # Highlight the 'Optimal Valley' with a pulsing green circle at E2
        valley_highlight = Circle(radius=0.5, color=GREEN_VALLEY, stroke_width=4)
        self.place_at_grid(valley_highlight, "E2")
        
        # Reposition valley_label to area F2-F3 to avoid bottom edge clipping
        valley_label = Text("Optimal Valley", font_size=20, color=GREEN_VALLEY)
        self.place_in_area(valley_label, "F2", "F3", scale_factor=0.8)

        self.play(
            FadeIn(valley_highlight),
            Write(valley_label)
        )
        
        # Path Arrow from C5 (peak) to E2 (valley)
        path_arrow = CurvedArrow(
            self.grid["C5"], self.grid["E2"], 
            angle=-TAU/8, color=WHITE, stroke_width=2
        )
        
        # Simple pulsing animation loop using succession for discrete keyframes
        self.play(
            valley_highlight.animate.scale(1.2),
            Create(path_arrow),
            run_time=0.8
        )
        self.play(
            valley_highlight.animate.scale(1/1.2),
            run_time=0.8
        )

        self.wait(2)
        
        # Reset colors for final state
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
