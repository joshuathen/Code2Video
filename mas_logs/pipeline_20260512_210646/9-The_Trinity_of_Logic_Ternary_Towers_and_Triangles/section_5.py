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
            "Ternary numbers act as addresses for every puzzle state.",
            "Solving the puzzle means traversing the fractal's edges.",
            "The longest path follows the triangle's outer perimeter."
        ]
        self.setup_layout("Fractal Traversal via Ternary Addresses", lecture_lines)

        # --- Data Generation ---
        def get_pos_normalized(addr):
            # Triangle corners (normalized)
            p0 = np.array([0, 1, 0])      # Top
            p1 = np.array([-0.866, -0.5, 0])  # BL
            p2 = np.array([0.866, -0.5, 0])   # BR
            
            curr_p0, curr_p1, curr_p2 = p0, p1, p2
            for char in addr:
                m01 = (curr_p0 + curr_p1) / 2
                m12 = (curr_p1 + curr_p2) / 2
                m20 = (curr_p2 + curr_p0) / 2
                if char == '0':
                    curr_p1, curr_p2 = m01, m20
                elif char == '1':
                    curr_p0, curr_p2 = m01, m12
                else: # '2'
                    curr_p0, curr_p1 = m20, m12
            return (curr_p0 + curr_p1 + curr_p2) / 3

        # Generate addresses for a 3-level Sierpinski triangle
        addresses = [np.base_repr(i, base=3).zfill(3) for i in range(27)]
        normalized_points = [get_pos_normalized(addr) for addr in addresses]
        
        # Dots representing states
        state_dots = VGroup(*[Dot(p, radius=0.03, color=GRAY) for p in normalized_points])
        # Issue 35: Fix overlap by moving to B2-E6 and scaling down
        self.place_in_area(state_dots, 'B2', 'E6', scale_factor=1.5)
        
        # Extract mapped positions
        real_points = [dot.get_center() for dot in state_dots]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Display the "Address System" label
        address_label = Text("Current Address:", font_size=24, color=WHITE)
        # Issue 34: Position at A2 with scale 0.8
        self.place_at_grid(address_label, "A2", scale_factor=0.8)
        
        # Initial counter
        counter = Text(addresses[0], font_size=32, color="#FFFF00")
        # Issue 34: Position at A5 with scale 0.8
        self.place_at_grid(counter, "A5", scale_factor=0.8)
        
        self.play(
            FadeIn(state_dots),
            Write(address_label),
            Write(counter)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF00")
        )
        
        # Trace the path along the fractal's edges
        path_lines = VGroup()
        for i in range(26):
            start_p = real_points[i]
            end_p = real_points[i+1]
            line = Line(start_p, end_p, color="#00FF00", stroke_width=5)
            path_lines.add(line)
            
            # Create updated counter text
            new_counter = Text(addresses[i+1], font_size=32, color="#FFFF00")
            # Maintain position at A5
            self.place_at_grid(new_counter, "A5", scale_factor=0.8)
            
            self.add(line)
            self.remove(counter)
            self.add(new_counter)
            counter = new_counter
            
            # Pace the traversal
            if i % 5 == 0:
                self.wait(0.1)
            else:
                self.wait(0.05)

        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # Corner markers
        corner_000 = Dot(real_points[0], radius=0.1, color=WHITE)
        corner_222 = Dot(real_points[-1], radius=0.1, color=WHITE)
        
        # Issue 28: Add Puzzle Icon asset
        puzzle_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/puzzle.svg").set_color(WHITE)
        solved_text = Text("Puzzle Solved", font_size=36, color="#FFFFFF")
        solved_group = VGroup(solved_text, puzzle_icon).arrange(RIGHT, buff=0.3)
        
        # Issue 36: Place at F3 with 0.9 scale
        self.place_at_grid(solved_group, "F3", scale_factor=0.9)

        self.play(
            Flash(corner_000, color=WHITE),
            Flash(corner_222, color=WHITE),
            FadeIn(solved_group)
        )
        
        self.wait(2)
