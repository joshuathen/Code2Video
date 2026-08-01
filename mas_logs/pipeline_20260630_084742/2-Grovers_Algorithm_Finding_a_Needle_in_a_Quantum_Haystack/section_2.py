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
        # Data
        title_text = "Prerequisite: Amplitudes and Superposition"
        lecture_lines = [
            'Quantum states are represented by probability amplitudes.',
            'The square of an amplitude determines its probability.',
            'We begin with an equal superposition of all possibilities.'
        ]
        
        # Initial Setup
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        CYAN = "#00FFFF"
        WHITE_COLOR = "#FFFFFF"
        YELLOW_COLOR = "#FFFF00"
        
        # Create Visual Components with Assets
        # 1. 8 Cyan bars [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/bars.svg]
        bars_svg = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/bars.svg")
        bars_svg.set_color(CYAN)
        
        # 2. Axis: White horizontal line
        axis = Line(start=LEFT * 2.5, end=RIGHT * 2.5, color=WHITE_COLOR)
        
        # 3. Labels |000> through |111> [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/labels.svg]
        labels_svg = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/labels.svg")
        labels_svg.set_color(WHITE_COLOR)
        
        # Align components relative to axis
        bars_svg.next_to(axis, UP, buff=0.1)
        labels_svg.next_to(axis, DOWN, buff=0.2)
            
        # Group components for positioning using the grid system
        chart_viz = VGroup(axis, bars_svg, labels_svg)
        # Fixed positioning as per VideoCritic (Issue 25)
        self.place_in_area(chart_viz, "A1", "E6", scale_factor=1.1)
        
        # Hide initial components for sequential animation
        axis.set_opacity(0)
        bars_svg.set_opacity(0)
        labels_svg.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.lecture[0].set_color(YELLOW_COLOR)
        # Draw axis and cyan bars representing amplitudes
        self.play(
            axis.animate.set_opacity(1),
            FadeIn(bars_svg, shift=UP),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight next line
        self.lecture[0].set_color(WHITE_COLOR)
        self.lecture[1].set_color(YELLOW_COLOR)
        # Show labels to define states; emphasize that height (amplitude) relates to prob
        self.play(
            FadeIn(labels_svg),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight final line
        self.lecture[1].set_color(WHITE_COLOR)
        self.lecture[2].set_color(YELLOW_COLOR)
        
        # Pulse effect: A white glow to represent equal superposition
        glow = bars_svg.copy().set_fill(WHITE_COLOR, opacity=0.3).set_stroke(WHITE_COLOR, width=2)
        self.add(glow)
        
        self.play(
            Indicate(bars_svg, color=WHITE_COLOR, scale_factor=1.1),
            glow.animate.scale(1.2).set_opacity(0),
            run_time=2,
            rate_func=slow_into
        )
        self.remove(glow)
        self.wait(3)
