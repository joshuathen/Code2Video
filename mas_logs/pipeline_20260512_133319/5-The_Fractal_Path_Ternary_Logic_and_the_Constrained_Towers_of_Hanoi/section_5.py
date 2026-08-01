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
        # Initializing layout with updated lecture lines
        lecture_lines = [
            "Solving the puzzle means traversing the fractal's outer perimeter.",
            "We travel from state 000 to state 222.",
            "This golden path follows the constrained move rules.",
            "Each step visits a new vertex along the edge.",
            "The solution touches every node on the boundary."
        ]
        self.setup_layout("Graph Traversal: Finding the Optimal Path", lecture_lines)

        # Assets
        disk_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/disk.svg").set_color(WHITE)
        puzzle_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/puzzle.svg").set_color(WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Build Sierpinski Gasket Depth 3
        def get_gasket_lines(depth, p1, p2, p3):
            if depth == 0:
                return [Line(p1, p2), Line(p2, p3), Line(p3, p1)]
            m12 = (p1 + p2) / 2
            m23 = (p2 + p3) / 2
            m31 = (p3 + p1) / 2
            return (get_gasket_lines(depth - 1, p1, m12, m31) +
                    get_gasket_lines(depth - 1, m12, p2, m23) +
                    get_gasket_lines(depth - 1, m31, m23, p3))

        rel_top = np.array([0, 2.5, 0])
        rel_bl = np.array([-2.5, -2.0, 0])
        rel_br = np.array([2.5, -2.0, 0])
        
        gasket_lines_raw = get_gasket_lines(3, rel_top, rel_bl, rel_br)
        gasket = VGroup(*gasket_lines_raw).set_stroke(color="#ADD8E6", width=1.5)
        
        # Position Gasket in B1-F5 area to avoid crowding title and right edge
        self.place_in_area(gasket, "B1", "F5", scale_factor=0.5)
        self.place_at_grid(disk_icon, "B6", scale_factor=0.4)
        
        self.play(Create(gasket), FadeIn(disk_icon), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        # Corner vertices for the labels and path
        g_top = gasket.get_top()
        g_left = gasket.get_corner(DL)
        g_right = gasket.get_corner(DR)
        
        label_000 = Text("000", font_size=20, color=WHITE)
        label_222 = Text("222", font_size=20, color=WHITE)
        
        label_000.next_to(g_top, UP, buff=0.1)
        label_222.next_to(g_right, DOWN, buff=0.1)
        
        self.play(Write(label_000), Write(label_222))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700") # Golden path
        
        # Path: Top -> Left (8 segments) then Left -> Right (8 segments)
        path_points = []
        for i in range(9):
            path_points.append(interpolate(g_top, g_left, i / 8))
        for i in range(1, 9):
            path_points.append(interpolate(g_left, g_right, i / 8))
            
        moving_dot = Dot(color="#FFD700").scale(1.2).move_to(path_points[0])
        self.add(moving_dot)
        
        # Start tracing Top -> Left
        traced_path = VGroup()
        for i in range(8):
            p_start = path_points[i]
            p_end = path_points[i+1]
            segment = Line(p_start, p_end, color="#FFD700", stroke_width=4)
            self.play(
                moving_dot.animate.move_to(p_end),
                Create(segment),
                run_time=0.15,
                rate_func=linear
            )
            traced_path.add(segment)
            
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(LIGHT_PINK)
        
        # Continue tracing Left -> Right with flashes at each vertex
        for i in range(8, len(path_points) - 1):
            p_start = path_points[i]
            p_end = path_points[i+1]
            segment = Line(p_start, p_end, color="#FFD700", stroke_width=4)
            
            self.play(
                moving_dot.animate.move_to(p_end),
                Create(segment),
                Flash(p_end, color=YELLOW, flash_radius=0.15, line_length=0.1),
                run_time=0.15,
                rate_func=linear
            )
            traced_path.add(segment)

        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#00FFFF")
        
        self.place_at_grid(puzzle_icon, "F6", scale_factor=0.5)
        
        # Add a glow effect to the entire path to show boundary coverage
        glow = traced_path.copy().set_stroke(width=10, opacity=0.3, color=GOLD)
        self.play(FadeIn(glow), FadeIn(puzzle_icon), run_time=1)
        
        self.wait(2)
