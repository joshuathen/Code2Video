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
        # Title and Lecture Lines
        lecture_lines = [
            "Classical searching checks unsorted items one by one.",
            "A million boxes might require half a million checks.",
            "Grover’s Algorithm finds the answer much faster.",
            "It needs only one thousand steps for that million.",
            "This is a massive quadratic speedup for searching."
        ]
        self.setup_layout("The Classical vs. Quantum Search Problem", lecture_lines)
        
        # Define Colors
        box_color = "#FFFFFF"
        scan_color = "#00FF00"
        star_color = "#FFD700"
        quantum_color = "#90EE90"
        
        # === Animation for Lecture Line 1 ===
        # Display four white boxes (#FFFFFF) labeled 1 to 4 in a row.
        self.play(self.lecture[0].animate.set_color(box_color))
        
        boxes = VGroup()
        for i in range(1, 5):
            sq = Square(side_length=0.8, color=box_color)
            lbl = Text(str(i), font_size=24, color=box_color)
            box = VGroup(sq, lbl)
            self.place_at_grid(box, f"B{i+1}")
            boxes.add(box)
            
        self.play(Create(boxes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A green circle (#00FF00) sequentially scans boxes 1 and 2.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(scan_color)
        )
        
        scanner = Circle(radius=0.15, color=scan_color, fill_opacity=1)
        # Start at Box 1
        self.place_at_grid(scanner, "B2")
        self.play(FadeIn(scanner))
        # Move scanner over Box 1 to Box 2
        self.play(scanner.animate.move_to(self.grid["B3"]), run_time=1)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # The green circle (#00FF00) scans box 3; it opens to reveal a gold star (#FFD700).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(star_color)
        )
        
        star = Star(n=5, outer_radius=0.3, color=star_color, fill_opacity=1)
        self.place_at_grid(star, "B4")
        
        # Scan to Box 3 and reveal star
        self.play(scanner.animate.move_to(self.grid["B4"]))
        self.play(
            boxes[2][1].animate.set_opacity(0), # Hide label '3'
            FadeIn(star)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display 'N = 1,000,000' and 'Classical: 500,000 checks' in white (#FFFFFF).
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(box_color)
        )
        
        comparison_c = Text("Classical: 500,000 checks", font_size=24, color=WHITE)
        # Fix: Centered in area D2 to D5
        self.place_in_area(comparison_c, 'D2', 'D5', scale_factor=0.8)
        
        self.play(Write(comparison_c))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Display 'Quantum: 1,000 steps' and 'O(sqrt(N))' in light green (#90EE90).
        # Summary of the speedup
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        comparison_q = Text("Quantum: 1,000 steps (O(sqrt(N)))", font_size=24, color=quantum_color)
        # Fix: Centered in area E2 to E5
        self.place_in_area(comparison_q, 'E2', 'E5', scale_factor=0.8)
        
        speedup_label = Text("Quadratic Speedup", font_size=32, color=YELLOW)
        # Fix: Centered in area F2 to F5 with higher visual hierarchy
        self.place_in_area(speedup_label, 'F2', 'F5', scale_factor=1.1)
        
        self.play(Write(comparison_q))
        self.play(FadeIn(speedup_label))
        self.wait(2)
        
        # Reset color
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
