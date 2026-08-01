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

class Section7Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Summary and Conclusion"
        lines = [
            "The Bridge connects rates of change with accumulation.",
            "Derivatives and integrals are two sides of one coin.",
            "This relationship is the heart of calculus."
        ]
        
        self.setup_layout(title, lines)
        
        # Define colors
        COLOR_SLOPE = "#FFFF00"
        COLOR_AREA = "#0000FF"
        COLOR_BRIDGE = "#FFD700"
        COLOR_TEXT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_SLOPE))
        
        # Create 'Slope' icon (Rate of Change)
        slope_icon = VGroup(
            Line(LEFT, RIGHT, color=GREY_B), # Small axis
            Line(0.5*DOWN + 0.5*LEFT, 0.5*UP + 0.5*RIGHT, color=COLOR_SLOPE, stroke_width=6) # Tangent line
        )
        slope_label = Text("Slope", font_size=24, color=COLOR_SLOPE)
        slope_group = VGroup(slope_icon, slope_label).arrange(DOWN, buff=0.2)
        
        # Fix: Move from B2 to B3 to avoid cramping the lecture notes (Issue 47)
        self.place_at_grid(slope_group, "B3", scale_factor=0.8)
        
        # Create 'Area' icon (Accumulation)
        area_icon = VGroup(
            Line(LEFT, RIGHT, color=GREY_B), # Small axis
            Polygon(
                0.5*LEFT, 0.5*RIGHT, 0.5*RIGHT + 0.4*UP, 0.5*LEFT + 0.7*UP,
                color=COLOR_AREA, fill_opacity=0.5, stroke_width=2
            )
        )
        area_label = Text("Area", font_size=24, color=COLOR_AREA)
        area_group = VGroup(area_icon, area_label).arrange(DOWN, buff=0.2)
        
        # Place at B5 to maintain separation
        self.place_at_grid(area_group, "B5", scale_factor=0.8)
        
        self.play(FadeIn(slope_group), FadeIn(area_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_BRIDGE)
        )
        
        # Draw a gold bridge connecting both icons using the SVG asset (Issue 34)
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg
        try:
            bridge_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
            bridge_svg.set_color(COLOR_BRIDGE)
            # Position it between the two groups (at B4)
            self.place_at_grid(bridge_svg, "B4", scale_factor=0.6)
        except Exception:
            # Fallback if asset is missing
            bridge_svg = ArcBetweenPoints(
                self.grid["B3"] + RIGHT*0.5, 
                self.grid["B5"] + LEFT*0.5, 
                angle=-TAU/8, 
                color=COLOR_BRIDGE, 
                stroke_width=6
            )
        
        bridge_text = Text("Calculus", font_size=24, color=COLOR_BRIDGE)
        # Place the text above the bridge (centered at B4 but shifted up)
        bridge_text.move_to(self.grid["B4"] + UP * 0.7)
        
        self.play(Create(bridge_svg), Write(bridge_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Pulse 'Fundamental Theorem of Calculus' text
        ftc_text = Text("Fundamental Theorem\nof Calculus", font_size=32, color=COLOR_TEXT)
        
        # Fix: Move from D2-E5 to D3-F5 to avoid lecture notes (Issue 48)
        self.place_in_area(ftc_text, "D3", "F5", scale_factor=0.8)
        
        self.play(FadeIn(ftc_text))
        # Use rate_functions.there_and_back (Belief B058)
        self.play(ftc_text.animate.scale(1.15), run_time=0.6, rate_func=rate_functions.there_and_back)
        self.play(ftc_text.animate.scale(1.15), run_time=0.6, rate_func=rate_functions.there_and_back)
        
        self.wait(2)
        
        # Final state
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(3)
