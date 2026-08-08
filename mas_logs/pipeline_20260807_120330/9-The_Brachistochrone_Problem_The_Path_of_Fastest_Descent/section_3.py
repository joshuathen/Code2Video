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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup the layout
        self.setup_layout("Prerequisite Knowledge: Snell's Law and Light", 
                         ["Light always follows the path of least time.", 
                          "Snell's Law describes how light bends between media.", 
                          "Light curves to spend more time in faster zones."])

        # Colors for elements
        ray_color = "#FFFFFF"
        snell_color = "#FFFF00"
        layer_color = "#888888"

        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg]
        light_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg").set_color(ray_color).scale(0.3)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(ray_color))
        
        # Build refraction diagram elements relative to origin
        interface = Line(LEFT * 2.5, RIGHT * 2.5, color=GRAY)
        normal = DashedLine(UP * 2, DOWN * 2, color=GRAY_A)
        
        p_hit = ORIGIN
        p_start = UP * 1.5 + LEFT * 1.5
        ray_part1 = Line(p_start, p_hit, color=ray_color)
        
        # Place light icon at start of ray
        light_icon_inst = light_icon.copy().move_to(p_start)
        
        # Refraction diagram group for initial placement (Issue 27)
        refraction_diag = VGroup(interface, normal, ray_part1, light_icon_inst)
        
        # Issue 27: Scale and position the diagram in B1-E6 to avoid cramping
        self.place_in_area(refraction_diag, 'B1', 'E6', scale_factor=0.85)
        
        self.play(Create(interface), Create(normal))
        self.play(Create(ray_part1), FadeIn(light_icon_inst))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(snell_color))
        
        # Second part of ray (refracted)
        # Reference position based on current interface center
        current_p_hit = interface.get_center()
        # Incident angle approx 45 deg relative to normal. Refracted angle approx 60 deg.
        p_end = current_p_hit + DOWN * 1.2 + RIGHT * 2.2
        ray_part2 = Line(current_p_hit, p_end, color=ray_color)
        
        # Angles
        arc1 = Arc(start_angle=PI/2, angle=ray_part1.get_angle() - PI/2, radius=0.4, arc_center=current_p_hit, color=snell_color)
        theta1 = MathTex(r"\theta_1", color=snell_color, font_size=24)
        theta1.next_to(arc1, UP + LEFT, buff=0.1)
        
        arc2 = Arc(start_angle=-PI/2, angle=ray_part2.get_angle() + PI/2, radius=0.4, arc_center=current_p_hit, color=snell_color)
        theta2 = MathTex(r"\theta_2", color=snell_color, font_size=24)
        theta2.next_to(arc2, DOWN + RIGHT, buff=0.1)
        
        # Snell's Law equation
        snell_eq = MathTex(r"n_1 \sin \theta_1 = n_2 \sin \theta_2", color=snell_color, font_size=32)
        # Issue 26: Move Snell's Law equation to A2-A5
        self.place_in_area(snell_eq, 'A2', 'A5')
        
        self.play(Create(ray_part2))
        self.play(Create(arc1), Write(theta1))
        self.play(Create(arc2), Write(theta2))
        self.play(Write(snell_eq))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(layer_color))
        
        # Transition out Snell's Law setup
        self.play(
            FadeOut(refraction_diag), FadeOut(ray_part2), FadeOut(arc1), FadeOut(theta1), 
            FadeOut(arc2), FadeOut(theta2), FadeOut(snell_eq)
        )
        
        # Simulate layered media (Issue 28)
        layered_medium = VGroup()
        for i in range(6):
            rect = Rectangle(width=5.5, height=0.9, color=layer_color, stroke_width=1)
            rect.set_fill(layer_color, opacity=0.05 + 0.08 * i)
            layered_medium.add(rect)
        layered_medium.arrange(DOWN, buff=0)
        
        # Issue 28: Fix layered_medium area to 'A1'-'F6'
        self.place_in_area(layered_medium, 'A1', 'F6')
        
        # Curved ray bending towards faster zones (lower refractive index)
        # Path starts from top-left and flattens as it goes down
        path_pts = [
            self.grid["A1"],
            self.grid["B2"],
            self.grid["C4"],
            self.grid["E6"]
        ]
        curved_ray = VMobject(color=ray_color)
        curved_ray.set_points_as_corners(path_pts)
        curved_ray.make_smooth()
        
        # Icon for movement
        light_icon_moving = light_icon.copy().move_to(path_pts[0])
        
        self.play(Create(layered_medium))
        self.play(FadeIn(light_icon_moving))
        self.play(
            Create(curved_ray),
            MoveAlongPath(light_icon_moving, curved_ray),
            run_time=4,
            rate_func=linear
        )
        self.wait(3)
