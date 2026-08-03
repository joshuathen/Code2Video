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
        # Setup layout
        title_text = "Real-World Application: Beyond Speed"
        lecture_lines = [
            "Derivatives measure change in more than just physical movement.",
            "They accurately predict power draw in your smartphone battery.",
            "Calculus is the language of our constantly changing world."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Asset path from Issue 24
        battery_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/battery.svg"

        # Initialization of variables
        # Using ValueTracker for the fill's width percentage
        fill_percent_tracker = ValueTracker(0.95) 

        # === Animation for Lecture Line 1 ===
        # A white (#FFFFFF) battery icon [Asset: battery.svg] outline appears on the screen.
        self.lecture[0].set_color("#FFFFFF")
        
        # Load SVG once (Mandatory: use asset)
        battery_svg = SVGMobject(battery_asset)
        battery_svg.set_color("#FFFFFF")
        # Ensure it acts as an outline
        battery_svg.set_fill(opacity=0)
        battery_svg.set_stroke(color=WHITE, width=4)
        
        # Resolve Issue 33: Position at B4 to E6 to avoid overlap with lecture notes
        self.place_in_area(battery_svg, "B4", "E6", scale_factor=1.5)
        
        self.play(Create(battery_svg), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A green (#00FF00) fill inside the battery slowly moves to the left.
        self.lecture[0].set_color("#888888")
        self.lecture[1].set_color("#00FF00")
        
        # Calculate fill dimensions relative to the SVG
        full_fill_width = battery_svg.width * 0.82
        fill_height = battery_svg.height * 0.75
        
        battery_fill = Rectangle(
            width=full_fill_width,
            height=fill_height,
            fill_color="#00FF00",
            fill_opacity=0.8,
            stroke_width=0
        )
        
        # Center and then shift left because of the battery tip
        battery_fill.move_to(battery_svg.get_center())
        battery_fill.shift(LEFT * (battery_svg.width * 0.05))
        
        # Persistent updater for stretching the fill based on the tracker
        battery_fill.add_updater(
            lambda m: m.stretch_to_fit_width(
                max(0.01, fill_percent_tracker.get_value() * full_fill_width), 
                about_edge=LEFT
            )
        )
        
        # Add fill and ensure the outline stays on top
        self.add(battery_fill)
        self.add_foreground_mobjects(battery_svg)
        
        # Slowly decrease the battery level (showing 'Power Draw' as rate of change)
        self.play(FadeIn(battery_fill, target_position=battery_svg.get_center()))
        self.play(fill_percent_tracker.animate.set_value(0.4), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A yellow (#FFFF00) arrow points to the decreasing edge, labeled 'Power Draw'.
        self.lecture[1].set_color("#888888")
        self.lecture[2].set_color("#FFFF00")
        
        # Create persistent arrow pointing downwards
        arrow = Arrow(
            start=UP * 0.7, 
            end=ORIGIN, 
            color="#FFFF00", 
            buff=0.1,
            stroke_width=5,
            max_tip_length_to_length_ratio=0.25
        )
        
        # Label created once outside updater for performance
        label = Text("Power Draw", font_size=20, color="#FFFF00")
        
        # Updater to keep arrow pointing to the moving right edge of the fill
        def update_arrow(m):
            # Calculate current right edge position
            fill_left_x = battery_fill.get_left()[0]
            current_width = fill_percent_tracker.get_value() * full_fill_width
            current_x = fill_left_x + current_width
            m.move_to(np.array([current_x, battery_svg.get_top()[1] + 0.5, 0]))

        arrow.add_updater(update_arrow)
        label.add_updater(lambda m: m.next_to(arrow, UP, buff=0.1))
        
        self.play(Create(arrow), Write(label))
        
        # Continue draining the battery while the arrow follows the changing derivative
        self.play(fill_percent_tracker.animate.set_value(0.08), run_time=3, rate_func=linear)
        self.wait(2)
