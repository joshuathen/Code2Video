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
        lines = ["Place points on a circle, connect every pair.",
                 "One point creates one slice.",
                 "Two points, two slices.",
                 "Three points, four slices.",
                 "Four points, eight slices."]
        self.setup_layout("The Hook: How many pieces?", lines)
        
        # Define base assets
        circle_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg"
        
        # Helper to create points on circle
        def get_circle_points(n, radius=1.0):
            return [np.array([np.cos(2*PI*i/n + PI/2), np.sin(2*PI*i/n + PI/2), 0]) * radius for i in range(n)]

        # --- Initial Setup ---
        circle = SVGMobject(circle_svg).set_color(WHITE)
        self.place_in_area(circle, 'C3', 'E5', scale_factor=1.5)
        self.add(circle)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        pt = Dot(point=circle.get_top(), color=RED)
        self.place_at_grid(pt, 'C4') # Position relative to circle center
        self.add(pt)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # 1 slice (already visualized by circle + 1 point)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        p2 = Dot(point=circle.get_bottom(), color=RED)
        chord = Line(pt.get_center(), p2.get_center(), color=YELLOW)
        self.add(p2, chord)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        pts = get_circle_points(3, radius=1.0)
        # Clear previous and redraw n=3 case
        self.remove(pt, p2, chord)
        dots = VGroup(*[Dot(point=p+circle.get_center(), color=RED) for p in pts])
        chords = VGroup(
            Line(pts[0]+circle.get_center(), pts[1]+circle.get_center(), color=YELLOW),
            Line(pts[1]+circle.get_center(), pts[2]+circle.get_center(), color=YELLOW),
            Line(pts[2]+circle.get_center(), pts[0]+circle.get_center(), color=YELLOW)
        )
        self.add(dots, chords)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        pts4 = get_circle_points(4, radius=1.0)
        # Clear and redraw n=4
        self.remove(dots, chords)
        dots4 = VGroup(*[Dot(point=p+circle.get_center(), color=RED) for p in pts4])
        # Internal chords
        c1 = Line(pts4[0]+circle.get_center(), pts4[2]+circle.get_center(), color=YELLOW)
        c2 = Line(pts4[1]+circle.get_center(), pts4[3]+circle.get_center(), color=YELLOW)
        outer = VGroup(
            Line(pts4[0]+circle.get_center(), pts4[1]+circle.get_center(), color=YELLOW),
            Line(pts4[1]+circle.get_center(), pts4[2]+circle.get_center(), color=YELLOW),
            Line(pts4[2]+circle.get_center(), pts4[3]+circle.get_center(), color=YELLOW),
            Line(pts4[3]+circle.get_center(), pts4[0]+circle.get_center(), color=YELLOW)
        )
        self.add(dots4, outer, c1, c2)
        self.wait(2)
