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
        lecture_lines = [
            'Riemann extended this function into the complex plane.', 
            "He mapped the zeros where the function's value vanishes.", 
            'These zeros align perfectly along a single vertical line.', 
            'Their rhythm defines the distribution of all prime numbers.', 
            'This landscape reveals the secret symmetry of the primes.'
        ]
        self.setup_layout("The Riemann Landscape: Finding Regularity", lecture_lines)
        
        # Hex Colors from prompt and descriptions
        COLOR_GRID = "#333333"
        COLOR_CRITICAL = "#00FF00"
        COLOR_ZERO = "#FF0000"
        COLOR_WAVE = "#00FFFF"
        COLOR_HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Coordinate grid representing the complex plane
        bg_lines = VGroup()
        for i in range(6):
            # Vertical lines (Cols 1-6)
            start_v = self.grid[f"A{i+1}"] + UP*0.5
            end_v = self.grid[f"F{i+1}"] + DOWN*0.5
            bg_lines.add(Line(start_v, end_v, color=COLOR_GRID, stroke_width=1))
            # Horizontal lines (Rows A-F)
            start_h = self.grid[f"{chr(65+i)}1"] + LEFT*0.5
            end_h = self.grid[f"{chr(65+i)}6"] + RIGHT*0.5
            bg_lines.add(Line(start_h, end_h, color=COLOR_GRID, stroke_width=1))
            
        re_label = Text("Real Axis", font_size=14, color=WHITE)
        im_label = Text("Imaginary Axis", font_size=14, color=WHITE)
        self.place_at_grid(re_label, 'F6', scale_factor=0.8)
        self.place_at_grid(im_label, 'A1', scale_factor=0.8)

        self.play(Create(bg_lines), Write(re_label), Write(im_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        # Zeros (red dots) appearing one by one
        zeros = VGroup(*[
            Dot(color=COLOR_ZERO, radius=0.12).move_to(self.grid[pos])
            for pos in ['B3', 'C3', 'D3', 'E3']
        ])
        
        zero_label = Text("Zeros (Valleys)", font_size=18, color=COLOR_ZERO)
        # Issue 40 fix: Positioning within area A5-A6
        self.place_in_area(zero_label, 'A5', 'A6', scale_factor=0.7)

        self.play(Write(zero_label))
        for zero in zeros:
            self.play(FadeIn(zero, scale=1.5), run_time=0.4)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Draw Critical Line at Re(s) = 1/2
        critical_line = Line(
            start=self.grid['A3'] + UP*0.5,
            end=self.grid['F3'] + DOWN*0.5,
            color=COLOR_CRITICAL,
            stroke_width=5
        )
        
        line_info = Text("Alignment = Regularity", font_size=18, color=COLOR_CRITICAL)
        # Issue 38 fix: Moved to grid B5
        self.place_at_grid(line_info, 'B5', scale_factor=0.8)

        self.play(Create(critical_line))
        self.play(Write(line_info))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        
        # Oscillating wave representing the 'music of primes'
        wave_points = []
        for i in range(21):
            alpha = i / 20
            x_base = self.grid['C5'][0]
            x_offset = 0.6 * np.sin(alpha * 6 * PI)
            y_val = self.grid['A5'][1] * (1 - alpha) + self.grid['F5'][1] * alpha
            wave_points.append(np.array([x_base + x_offset, y_val, 0]))
            
        wave = VMobject(color=COLOR_WAVE)
        wave.set_points_as_corners(wave_points).make_smooth()
        
        wave_text = Text("Prime Music", font_size=18, color=COLOR_WAVE)
        # Issue 39 fix: Moved to grid B6
        self.place_at_grid(wave_text, 'B6', scale_factor=0.8)
        
        # Issue 25: Load and place instrument icon
        instrument_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/instrument.svg")
        self.place_at_grid(instrument_icon, 'A6', scale_factor=0.4)

        self.play(Create(wave), Write(wave_text), FadeIn(instrument_icon), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        # Flash the Critical Line
        self.play(
            critical_line.animate.set_stroke(width=12),
            Flash(critical_line, color=COLOR_CRITICAL, line_length=0.4, num_lines=15),
            run_time=0.8
        )
        self.play(critical_line.animate.set_stroke(width=5), run_time=0.5)
        
        # Symmetry arrows (Revealing spacing logic)
        spacing_arrows = VGroup()
        for i in range(len(zeros)-1):
            arrow = DoubleArrow(
                zeros[i].get_center(), 
                zeros[i+1].get_center(), 
                buff=0.1, 
                color=WHITE, 
                stroke_width=2,
                tip_length=0.15
            )
            spacing_arrows.add(arrow)
            
        self.play(Create(spacing_arrows))
        self.wait(2)
        
        # Final cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(2)
