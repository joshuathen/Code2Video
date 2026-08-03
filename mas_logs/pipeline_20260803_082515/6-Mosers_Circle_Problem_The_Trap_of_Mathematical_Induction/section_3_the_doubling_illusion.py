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

class Section3TheDoublingIllusionScene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "One point leaves the pizza as one whole piece.",
            "Two points and one cut create two regions.",
            "Three points connected create four total regions.",
            "Four points yield eight regions after all cuts.",
            "Five points create sixteen regions. The pattern doubles!"
        ]
        self.setup_layout("The Alluring Pattern: n = 1 to 5", lecture_lines)

        # Define Colors
        COLOR_N1 = "#FFFFFF"
        COLOR_N2 = "#FFFFFF"
        COLOR_N3 = "#FFFFFF"
        COLOR_N4 = "#FFFFFF"
        COLOR_TABLE = "#00FFFF"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_PATTERN = "#FFD700"

        # Circle setup
        circle = Circle(radius=1.8, color=WHITE)
        self.place_in_area(circle, "A1", "D4")
        self.add(circle)

        # Pre-calculate points for n=5
        # Points on the circle
        angles = [90, 20, 200, 320, 140] # Degrees
        points_pos = [circle.point_at_angle(a * DEGREES) for a in angles]
        
        # Table setup
        table_header_n = Text("n", font_size=22, color=WHITE)
        table_header_r = Text("R", font_size=22, color=WHITE)
        
        table_vals_n = VGroup(*[Text(str(i), font_size=20, color=WHITE) for i in range(1, 6)])
        table_vals_r = VGroup(*[Text(str(2**(i-1)), font_size=20, color=WHITE) for i in range(1, 6)])
        
        # Position headers
        self.place_at_grid(table_header_n, "A5")
        self.place_at_grid(table_header_r, "A6")
        self.add(table_header_n, table_header_r)

        # Position data rows in B5-F5 (n) and B6-F6 (R)
        for i in range(5):
            row_label = chr(ord('B') + i)
            self.place_at_grid(table_vals_n[i], f"{row_label}5")
            self.place_at_grid(table_vals_r[i], f"{row_label}6")

        # Label for regions in the circle
        # FIXED: Issue 30 - use 'E2' to avoid overlap
        label_r = Text("R=1", font_size=24, color=WHITE)
        self.place_at_grid(label_r, "E2", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_N1)
        p1 = Dot(points_pos[0], color=COLOR_N1)
        self.play(Create(p1), Write(label_r))
        self.play(Write(table_vals_n[0]), Write(table_vals_r[0]))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_N2)
        p2 = Dot(points_pos[1], color=COLOR_N2)
        c12 = Line(points_pos[0], points_pos[1], color=COLOR_N2)
        label_r_2 = Text("R=2", font_size=24, color=COLOR_N2)
        self.place_at_grid(label_r_2, "E2", scale_factor=0.8)

        self.play(
            Create(p2),
            Create(c12),
            Transform(label_r, label_r_2)
        )
        self.play(Write(table_vals_n[1]), Write(table_vals_r[1]))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_N3)
        p3 = Dot(points_pos[2], color=COLOR_N3)
        c13 = Line(points_pos[0], points_pos[2], color=COLOR_N3)
        c23 = Line(points_pos[1], points_pos[2], color=COLOR_N3)
        label_r_3 = Text("R=4", font_size=24, color=COLOR_N3)
        self.place_at_grid(label_r_3, "E2", scale_factor=0.8)

        self.play(
            Create(p3),
            Create(VGroup(c13, c23)),
            Transform(label_r, label_r_3)
        )
        self.play(Write(table_vals_n[2]), Write(table_vals_r[2]))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_N4)
        p4 = Dot(points_pos[3], color=COLOR_N4)
        chords4 = VGroup(*[Line(points_pos[i], points_pos[3], color=COLOR_N4) for i in range(3)])
        label_r_4 = Text("R=8", font_size=24, color=COLOR_N4)
        self.place_at_grid(label_r_4, "E2", scale_factor=0.8)

        self.play(
            Create(p4),
            Create(chords4),
            Transform(label_r, label_r_4)
        )
        self.play(Write(table_vals_n[3]), Write(table_vals_r[3]))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_TABLE)
        p5 = Dot(points_pos[4], color=COLOR_TABLE)
        chords5 = VGroup(*[Line(points_pos[i], points_pos[4], color=COLOR_TABLE) for i in range(4)])
        label_r_5 = Text("R=16", font_size=24, color=COLOR_TABLE)
        self.place_at_grid(label_r_5, "E2", scale_factor=0.8)

        self.play(
            Create(p5),
            Create(chords5),
            Transform(label_r, label_r_5)
        )
        self.play(Write(table_vals_n[4]), Write(table_vals_r[4]))
        
        # Highlighting the doubling pattern
        highlight_boxes = VGroup(*[
            SurroundingRectangle(table_vals_r[i], color=COLOR_HIGHLIGHT, buff=0.1)
            for i in range(5)
        ])
        
        doubling_arrows = VGroup()
        x2_labels = VGroup()
        for i in range(4):
            # Curved arrows between the values in the R column
            start_p = table_vals_r[i].get_right()
            end_p = table_vals_r[i+1].get_right()
            arrow = CurvedArrow(start_p, end_p, angle=-PI/2, color=COLOR_HIGHLIGHT).scale(0.6)
            doubling_arrows.add(arrow)
            x2 = Text("×2", font_size=16, color=COLOR_HIGHLIGHT).next_to(arrow, RIGHT, buff=0.1)
            x2_labels.add(x2)

        self.play(Create(highlight_boxes))
        self.play(Create(doubling_arrows), Write(x2_labels))
        
        # Label the pattern formula
        # FIXED: Issue 31 - use place_in_area 'F3', 'F4' to center under table
        pattern_formula = MathTex("R = 2^{n-1}", font_size=36, color=COLOR_PATTERN)
        self.place_in_area(pattern_formula, 'F3', 'F4', scale_factor=0.8)
        self.play(Write(pattern_formula))
        self.wait(2)
