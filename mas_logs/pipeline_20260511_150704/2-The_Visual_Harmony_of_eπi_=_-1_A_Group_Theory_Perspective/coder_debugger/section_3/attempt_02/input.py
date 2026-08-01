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

        # Define fine-grained animation grid (6x6 grid on right side)
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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Group Theory Lens: Symmetry and SO(2)", 
            [
                "All unit rotations form the group called U(1).", 
                "Addition in exponents maps to multiplication on the circle.", 
                "Group theory views this formula as a symmetry map."
            ]
        )
        
        # Colors
        GREY_CIRCLE = "#888888"
        LIGHT_BLUE = "#ADD8E6"
        HIGHLIGHT_COLOR = YELLOW
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # 1. Fade in a unit circle in grey
        circle = Circle(radius=1.5, color=GREY_CIRCLE)
        self.place_in_area(circle, 'B2', 'E5')
        center = circle.get_center()
        
        # To show rotation, add a radius line and a dot
        radius_line = Line(center, circle.point_at_angle(0), color=WHITE, stroke_width=2)
        rotation_dot = Dot(circle.point_at_angle(0), color=WHITE, radius=0.05)
        
        group_visual = VGroup(circle, radius_line, rotation_dot)
        
        # 3. Display text 'Group SO(2)'
        so2_label = Text("Group SO(2)", color=LIGHT_BLUE, font_size=24)
        self.place_at_grid(so2_label, 'A3', scale_factor=1.2)
        
        self.play(
            FadeIn(circle),
            FadeIn(so2_label),
            FadeIn(radius_line),
            FadeIn(rotation_dot)
        )
        
        # 2. Show a rotation animation (continuous group action)
        rotation_tracker = ValueTracker(0)
        
        def update_group(m):
            angle = rotation_tracker.get_value()
            target_point = circle.point_at_angle(angle)
            m[1].put_start_and_end_on(center, target_point)
            m[2].move_to(target_point)
            
        group_visual.add_updater(update_group)
        
        self.play(rotation_tracker.animate.set_value(2 * PI), run_time=3, rate_func=linear)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # 4. Draw a mapping arrow from the Real number line
        real_line = NumberLine(
            x_range=[-2, 2, 1],
            length=3,
            color=WHITE,
            include_numbers=True,
            font_size=18,
            label_constructor=Text
        )
        self.place_at_grid(real_line, 'F3')
        real_label = Text("Real Numbers (Additive)", font_size=16, color=WHITE)
        self.place_at_grid(real_label, 'F5')
        
        # Stop continuous rotation for demonstration
        group_visual.remove_updater(update_group)
        
        # Arrow from number line to circle
        map_arrow = CurvedArrow(
            real_line.n2p(1), 
            circle.point_at_angle(1), 
            angle=-PI/2, 
            color=YELLOW,
            tip_length=0.15
        )
        
        self.play(Create(real_line), FadeIn(real_label))
        self.play(Create(map_arrow))
        self.wait(1)
        
        # 5. Highlight addition of two angles
        theta1 = 1.0
        theta2 = 0.8
        
        arc1 = Arc(radius=1.5, start_angle=0, angle=theta1, color=BLUE, arc_center=center)
        arc2 = Arc(radius=1.5, start_angle=theta1, angle=theta2, color=GREEN, arc_center=center)
        
        self.play(Create(arc1))
        self.play(Create(arc2))
        
        formula_text = Text("exp(ia) * exp(ib) = exp(i(a+b))", font_size=18, color=HIGHLIGHT_COLOR)
        self.place_at_grid(formula_text, 'E3', scale_factor=1.0)
        self.play(Write(formula_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        symmetry_text = Text("Symmetry: Rotation Transformation", font_size=18, color=LIGHT_BLUE)
        self.place_at_grid(symmetry_text, 'D3', scale_factor=1.0)
        
        self.play(
            FadeOut(arc1), FadeOut(arc2), FadeOut(map_arrow), FadeOut(formula_text),
            FadeIn(symmetry_text)
        )
        
        # Resume rotation using the dot/line to represent symmetry in action
        group_visual.add_updater(update_group)
        self.play(rotation_tracker.animate.set_value(4 * PI), run_time=4, rate_func=linear)
        self.wait(2)