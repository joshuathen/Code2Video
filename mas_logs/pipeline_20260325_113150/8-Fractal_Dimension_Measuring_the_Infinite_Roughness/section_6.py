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
        # Setup layout with title and lecture lines
        self.setup_layout("Real-world Application: Nature's Design", [
            "Lungs use fractal branching to maximize surface.",
            "Fractal antennas capture many frequencies efficiently.",
            "Nature uses roughness to solve complex problems."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line with matching color
        self.play(self.lecture[0].animate.set_color("#FFCCCC"))
        
        # Recursive function to create lung-like branching
        def create_lung_tree(point, length, angle, depth):
            if depth == 0:
                return VGroup()
            end_p = point + np.array([length * np.cos(angle), length * np.sin(angle), 0])
            line = Line(point, end_p, color="#FFCCCC", stroke_width=depth * 0.7)
            group = VGroup(line)
            # Branching factor and angles to simulate natural structure
            group.add(create_lung_tree(end_p, length * 0.75, angle + 0.45, depth - 1))
            group.add(create_lung_tree(end_p, length * 0.75, angle - 0.45, depth - 1))
            return group

        # Compose lung structure (trachea + two lobes)
        trachea = Line(UP * 0.5, ORIGIN, color="#FFCCCC", stroke_width=7)
        left_lobe = create_lung_tree(ORIGIN, 0.6, 0.7 * PI, 6)
        right_lobe = create_lung_tree(ORIGIN, 0.6, 0.3 * PI, 6)
        lungs = VGroup(trachea, left_lobe, right_lobe)
        
        # Use grid system for placement (Issue 59 Fix: repositioned to A1-C3)
        self.place_in_area(lungs, "A1", "C3", scale_factor=0.7)
        
        # Add label (Issue 59 Fix: repositioned to B4 for proximity and visibility)
        lung_label = Text("Lungs (D ≈ 2.9): Max Area", font_size=18, color=WHITE)
        self.place_at_grid(lung_label, "B4", scale_factor=0.8)

        self.play(Create(lungs), Write(lung_label), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to second line with matching color
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#AAAAAA")
        )

        # Recursive function to create a fractal antenna (H-fractal pattern)
        def create_antenna_fractal(center, length, depth):
            if depth == 0:
                return VGroup()
            
            # Draw 'H' unit
            h_bar = Line(center + LEFT*length/2, center + RIGHT*length/2, color="#AAAAAA", stroke_width=max(1, depth))
            v_l = Line(center + LEFT*length/2 + UP*length/2, center + LEFT*length/2 + DOWN*length/2, color="#AAAAAA", stroke_width=max(1, depth))
            v_r = Line(center + RIGHT*length/2 + UP*length/2, center + RIGHT*length/2 + DOWN*length/2, color="#AAAAAA", stroke_width=max(1, depth))
            h_unit = VGroup(h_bar, v_l, v_r)
            
            # Recursively add smaller H's to the ends
            new_len = length / 2
            h_unit.add(create_antenna_fractal(center + LEFT*length/2 + UP*length/2, new_len, depth - 1))
            h_unit.add(create_antenna_fractal(center + LEFT*length/2 + DOWN*length/2, new_len, depth - 1))
            h_unit.add(create_antenna_fractal(center + RIGHT*length/2 + UP*length/2, new_len, depth - 1))
            h_unit.add(create_antenna_fractal(center + RIGHT*length/2 + DOWN*length/2, new_len, depth - 1))
            return h_unit

        # Create antenna and smartphone icon
        antenna = create_antenna_fractal(ORIGIN, 1.0, 4)
        phone_body = RoundedRectangle(corner_radius=0.1, height=1.6, width=0.8, color=WHITE, stroke_width=2)
        phone_screen = Rectangle(height=1.2, width=0.7, color=WHITE, stroke_width=1).move_to(phone_body.get_center())
        smartphone = VGroup(phone_body, phone_screen)
        
        # Combine and position (Issue 60 Fix: repositioned to D1-F3)
        antenna_system = VGroup(antenna, smartphone).arrange(RIGHT, buff=0.4)
        self.place_in_area(antenna_system, "D1", "F3", scale_factor=0.8)
        
        # Add label (Issue 60 Fix: repositioned to E4 for proximity)
        antenna_label = Text("Antenna: Multi-band", font_size=18, color=WHITE)
        self.place_at_grid(antenna_label, "E4", scale_factor=0.9)

        self.play(FadeIn(antenna_system), Write(antenna_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight concluding summary line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Final visualization: emphasize complexity/efficiency
        self.play(
            lungs.animate.scale(1.1),
            antenna_system.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
