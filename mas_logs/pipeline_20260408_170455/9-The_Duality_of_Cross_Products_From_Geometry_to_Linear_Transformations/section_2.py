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
        # Initialize the layout
        title_str = "Geometric Interpretation: Magnitude and Direction"
        lecture_content = [
            "The cross product's magnitude equals the parallelogram's area.",
            "The resulting vector is perpendicular to the inputs.",
            "Use the right-hand rule to find the direction.",
            "Point fingers from vector A towards vector B.",
            "Your thumb reveals the cross product's orientation."
        ]
        self.setup_layout(title_str, lecture_content)

        # Colors from prompt
        COLOR_A = "#FF5733"
        COLOR_B = "#33FF57"
        COLOR_CP = "#3357FF"
        COLOR_AREA = "#FFFF33"

        # Coordinate System setup
        axes = Axes(
            x_range=[-1, 3, 1],
            y_range=[-1, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY_B}
        )
        # Apply slight skew for perspective feel as requested ("3D-perspective vectors")
        axes.apply_matrix([[1, 0.2, 0], [0, 1, 0], [0, 0, 1]]).rotate(-10 * DEGREES)
        self.place_in_area(axes, "A3", "F6", scale_factor=1.1)
        
        origin = axes.get_origin()

        # Vectors
        vec_a = Arrow(origin, axes.coords_to_point(2, 0.5, 0), buff=0, color=COLOR_A)
        vec_b = Arrow(origin, axes.coords_to_point(0.5, 2, 0), buff=0, color=COLOR_B)
        
        # Labels for vectors - Issue 30 fixes
        label_a = Text("a", font_size=24, color=COLOR_A)
        self.place_at_grid(label_a, "C6", scale_factor=0.8)
        
        label_b = Text("b", font_size=24, color=COLOR_B)
        self.place_at_grid(label_b, "B4", scale_factor=0.8)

        # Parallelogram and Area Label
        poly_points = [
            origin,
            axes.coords_to_point(2, 0.5, 0),
            axes.coords_to_point(2.5, 2.5, 0),
            axes.coords_to_point(0.5, 2, 0)
        ]
        para = Polygon(*poly_points, fill_opacity=0.3, fill_color=COLOR_AREA, stroke_color=COLOR_AREA)
        area_text = Text("|a x b|", font_size=18, color=COLOR_AREA)
        area_text.move_to(para.get_center())

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_AREA))
        self.play(Create(axes))
        self.play(GrowArrow(vec_a), Write(label_a))
        self.play(GrowArrow(vec_b), Write(label_b))
        self.play(Create(para), FadeIn(area_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(COLOR_CP))
        
        # Cross product vector (simulated as vertical in perspective)
        cp_end = origin + UP * 2.5
        vec_cp = Arrow(origin, cp_end, buff=0, color=COLOR_CP)
        label_cp = Text("a x b", font_size=24, color=COLOR_CP).next_to(cp_end, UP, buff=0.1)
        
        # Right angle indicators relative to origin
        r_angle1 = Line(origin + RIGHT*0.15, origin + RIGHT*0.15 + UP*0.15, color=COLOR_CP, stroke_width=2)
        r_angle2 = Line(origin + UP*0.15, origin + RIGHT*0.15 + UP*0.15, color=COLOR_CP, stroke_width=2)
        ortho_marker = VGroup(r_angle1, r_angle2)

        self.play(GrowArrow(vec_cp), Write(label_cp))
        self.play(Create(ortho_marker))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(WHITE))
        # Abstract right hand graphic (circle representing palm at origin)
        hand_icon = Circle(radius=0.4, color=WHITE, stroke_width=2).move_to(origin)
        self.play(Create(hand_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(WHITE))
        # Visualize finger pointing direction
        finger_sweep = Arc(radius=0.7, start_angle=vec_a.get_angle(), angle=vec_b.get_angle() - vec_a.get_angle(), arc_center=origin, color=WHITE)
        self.play(Create(finger_sweep), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(COLOR_CP))
        # Pulse the resulting vector to show orientation result
        self.play(vec_cp.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        self.wait(2)
