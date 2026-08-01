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
        # Initial layout setup with 5 lecture lines
        lecture_lines = [
            'This formula links any two coordinate systems together.',
            'Basis vectors define how we measure the world.',
            'Changing basis is like looking through a different lens.',
            'The world stays fixed; only our description changes.',
            'Ultimately, basis is just a choice of language.'
        ]
        self.setup_layout("Summary and Intuition Wrap-up", lecture_lines)
        
        # Matrix P for basis transformation
        matrix_p = [[1.2, 0.4, 0], [0.3, 0.9, 0], [0, 0, 1]]

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#58C4DD"), run_time=0.5)
        
        # Formula: [v]_std = P * [v]_B
        formula = Text("[v]_std = P * [v]_B", color=WHITE, font_size=28)
        # Issue 44: Positioning formula with scale_factor=0.8
        self.place_in_area(formula, 'A1', 'A6', scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#83C167"),
            run_time=0.5
        )
        
        # Setup standard grid
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            background_line_style={
                "stroke_color": "#666666",
                "stroke_width": 2,
                "stroke_opacity": 0.6
            },
            axis_config={"stroke_color": "#888888"}
        )
        # Issue 45: Position plane with scale_factor=0.8
        self.place_in_area(plane, 'B1', 'E6', scale_factor=0.8)
        plane.save_state()
        
        # Vectors fixed in world space
        origin = plane.get_center()
        vec_red = Vector([1.0, 1.0], color="#FC6255").shift(origin)
        vec_blue = Vector([-1.2, 0.6], color="#58C4DD").shift(origin)
        
        # Asset: world.svg (Issue 31)
        world_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/world.svg")
        self.place_at_grid(world_icon, 'B6', scale_factor=0.3)
        
        self.play(
            Create(plane),
            GrowFromPoint(vec_red, origin),
            GrowFromPoint(vec_blue, origin),
            FadeIn(world_icon),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # Morph standard grid into slanted grid
        self.play(
            plane.animate.apply_matrix(matrix_p, about_point=origin),
            run_time=2.5,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight Line 4
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FF8080"),
            run_time=0.5
        )
        
        # Morph grid back to standard, vectors stay stationary
        self.play(
            Restore(plane),
            run_time=2.0,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight Line 5
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FC6255"),
            run_time=0.5
        )
        
        # Final takeaway text
        takeaway = Text("Basis is just a choice of language", color="#FFFF00", font_size=28)
        # Issue 46: Position takeaway with scale_factor=0.8
        self.place_in_area(takeaway, 'F1', 'F6', scale_factor=0.8)
        
        # Asset: lens.svg (Issue 31)
        lens_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/lens.svg")
        self.place_at_grid(lens_icon, 'F6', scale_factor=0.3)
        
        self.play(
            Write(takeaway),
            FadeIn(lens_icon),
            run_time=1.5
        )
        self.wait(3)
