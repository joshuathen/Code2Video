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
        # Initial layout setup
        self.setup_layout(
            "The Proof Logic: Intersection of Surfaces", 
            [
                "The strip's boundary is our original 2D curve.",
                "Imagine a flat plane slicing through this strip.",
                "Topology proves the strip must intersect this plane.",
                "These intersections reveal sets of four special points.",
                "These points are the corners of our hidden square."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        # 2D Möbius strip projection boundary
        mobius_boundary = ParametricFunction(
            lambda t: np.array([
                (1.2 + 0.4 * np.cos(t / 2)) * np.cos(t),
                (1.2 + 0.4 * np.cos(t / 2)) * np.sin(t),
                0
            ]),
            t_range=[0, 4 * PI],
            color="#FFFFFF",
            stroke_width=4
        )
        
        # Create a light fill to represent the surface projection
        mobius_fill = Polygon(
            *[mobius_boundary.point_from_proportion(alpha) for alpha in np.linspace(0, 1, 100)],
            fill_color="#FFFFFF", 
            fill_opacity=0.1, 
            stroke_width=0
        )
        
        strip_group = VGroup(mobius_fill, mobius_boundary)
        # Position the strip projection in the lower area
        self.place_in_area(strip_group, "C2", "F5", scale_factor=0.8)
        
        self.play(Create(mobius_boundary), FadeIn(mobius_fill), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        
        # Horizontal plane represented as a yellow line slicing the strip
        h_line = Line(
            self.grid["D1"], 
            self.grid["D6"], 
            color="#FFFF00", 
            stroke_width=3
        )
        
        self.play(Create(h_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        
        # Issue 55: Four symmetrical intersection points at D2, D3, D4, D5
        dots = VGroup(*[Circle(radius=0.1, color="#FF0000", fill_opacity=1) for _ in range(4)])
        self.place_at_grid(dots[0], "D2")
        self.place_at_grid(dots[1], "D3")
        self.place_at_grid(dots[2], "D4")
        self.place_at_grid(dots[3], "D5")
        
        self.play(FadeIn(dots, lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        
        # Mapping back to the original curve logic
        orig_loop = Circle(radius=0.7, color="#FFFFFF").set_stroke(opacity=0.4)
        rectangle_peg = Rectangle(width=1.0, height=0.6, color="#00FF00").rotate(0.3)
        loop_container = VGroup(orig_loop, rectangle_peg)
        
        # Issue 56: Position loop_container at A2-B3, scale 0.8
        self.place_in_area(loop_container, 'A2', 'B3', scale_factor=0.8)
        
        # Visual connector from the intersection set to the mapped curve
        mapping_arrow = Arrow(
            self.grid["D3"], 
            self.grid["B3"], 
            color="#FFFFFF", 
            buff=0.2,
            stroke_width=2
        )
        
        self.play(
            Create(orig_loop), 
            Create(rectangle_peg), 
            GrowArrow(mapping_arrow)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        
        # Morphing the found rectangle into the final target square
        square_peg = Square(side_length=0.8, color="#FFFF00").rotate(0.5)
        square_peg.move_to(rectangle_peg.get_center())
        
        # Issue 57: Question mark at B4 to indicate the specific square condition
        q_mark = Text("?", color="#FFFF00", font_size=32)
        self.place_at_grid(q_mark, 'B4', scale_factor=0.8)
        
        self.play(
            Transform(rectangle_peg, square_peg),
            Write(q_mark),
            run_time=1.5
        )
        self.wait(2)
