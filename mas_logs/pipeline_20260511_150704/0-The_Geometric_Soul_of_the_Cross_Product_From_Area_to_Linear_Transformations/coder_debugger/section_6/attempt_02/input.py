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

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Cross products link area, direction, and 3D determinants.',
            'They describe physical rotations and torque on objects.',
            'Vector-Bot orbits smoothly guided by the Lorentz Force.'
        ]
        self.setup_layout("Synthesis and Real-World Application", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Geometric visual: Parallelogram and Determinant
        p_origin = self.grid['D1']
        p_v1 = np.array([1.2, 0.4, 0])
        p_v2 = np.array([0.4, 1.2, 0])
        parallelogram = Polygon(
            p_origin, p_origin + p_v1, p_origin + p_v1 + p_v2, p_origin + p_v2,
            fill_opacity=0.6, fill_color=TEAL, stroke_color=WHITE
        )
        
        # Pseudo-normal vector to represent direction
        normal_start = parallelogram.get_center()
        normal_vector = Arrow(normal_start, normal_start + np.array([0.5, 0.5, 0]) * 1.5, color=GOLD, buff=0)
        # Replaced MathTex with Text to avoid LaTeX dependency error
        normal_label = Text("n", color=GOLD, font_size=20).next_to(normal_vector.get_end(), UR, buff=0.1)
        
        # Determinant Formula - Replaced MathTex with Text to avoid LaTeX dependency error
        det_tex = Text("det(Matrix)", font_size=24, color=WHITE)
        self.place_in_area(det_tex, "B4", "D6")
        
        group1 = VGroup(parallelogram, normal_vector, normal_label, det_tex)
        self.play(FadeIn(group1))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(FadeOut(group1))
        
        # Torque representation: tau = r x F
        pivot_point = self.grid['D3']
        lever_arm_line = Line(pivot_point, pivot_point + RIGHT * 2, color=GREY_B, stroke_width=6)
        pivot_dot = Dot(pivot_point, color=WHITE)
        
        r_vec = Arrow(pivot_point, pivot_point + RIGHT * 2, color=BLUE, buff=0)
        f_vec = Arrow(pivot_point + RIGHT * 2, pivot_point + RIGHT * 2 + UP * 1.5, color=RED, buff=0)
        
        # Replaced MathTex with Text to avoid LaTeX dependency error
        r_label = Text("r", color=BLUE, font_size=24).next_to(r_vec, DOWN, buff=0.1)
        f_label = Text("F", color=RED, font_size=24).next_to(f_vec, RIGHT, buff=0.1)
        torque_formula = Text("τ = r × F", font_size=30).move_to(self.grid['B3'])
        
        group2 = VGroup(lever_arm_line, pivot_dot, r_vec, f_vec, r_label, f_label, torque_formula)
        
        self.play(Create(lever_arm_line), FadeIn(pivot_dot))
        self.play(GrowArrow(r_vec), Write(r_label))
        self.play(GrowArrow(f_vec), Write(f_label))
        self.play(Write(torque_formula))
        
        # Rotation hint
        arc = Arc(radius=0.5, start_angle=0, angle=PI/2, arc_center=pivot_point + RIGHT*2, color=YELLOW).add_tip()
        self.play(Create(arc))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(FadeOut(group2), FadeOut(arc))
        
        # Lorentz Force: F = q(v x B)
        center_orbit = self.grid['C4']
        orbit_radius = 1.3
        orbit_path = Circle(radius=orbit_radius, color=WHITE, stroke_opacity=0.15).move_to(center_orbit)
        self.add(orbit_path)
        
        # Magnetic Field B (Blue crosses into page)
        b_field_color = "#0000FF"
        b_field = VGroup()
        for r_code in ["B", "C", "D"]:
            for c_code in ["2", "3", "4", "5", "6"]:
                # Replaced MathTex with Text to avoid LaTeX dependency error
                cross = Text("×", color=b_field_color, font_size=18)
                self.place_at_grid(cross, f"{r_code}{c_code}")
                b_field.add(cross)
        
        b_label = Text("B Field (into page)", font_size=16, color=b_field_color).move_to(self.grid['F4'])
        
        # Vector-Bot
        bot_body = Circle(radius=0.18, fill_opacity=1, color=WHITE, stroke_color=GREY)
        bot_eye_l = Dot(radius=0.03, color=BLACK).move_to(bot_body.get_center() + LEFT*0.06 + UP*0.06)
        bot_eye_r = Dot(radius=0.03, color=BLACK).move_to(bot_body.get_center() + RIGHT*0.06 + UP*0.06)
        vector_bot = VGroup(bot_body, bot_eye_l, bot_eye_r)
        vector_bot.move_to(center_orbit + RIGHT * orbit_radius)
        
        # Vectors v (#52CEFF) and F (#FF00FF)
        v_color = "#52CEFF"
        f_color = "#FF00FF"
        
        v_arrow = Arrow(vector_bot.get_center(), vector_bot.get_center() + UP, color=v_color, buff=0)
        f_arrow = Arrow(vector_bot.get_center(), vector_bot.get_center() + LEFT, color=f_color, buff=0)
        
        # Replaced MathTex with Text to avoid LaTeX dependency error
        v_tag = Text("v", color=v_color, font_size=22)
        f_tag = Text("F", color=f_color, font_size=22)
        
        # Updaters for orbital motion
        self.orbit_angle = 0
        
        def update_bot(m, dt):
            self.orbit_angle += dt * 1.2
            new_pos = center_orbit + np.array([
                np.cos(self.orbit_angle) * orbit_radius,
                np.sin(self.orbit_angle) * orbit_radius,
                0
            ])
            m.move_to(new_pos)
            
        def update_v(m):
            pos = vector_bot.get_center()
            rel = pos - center_orbit
            tangent = np.array([-rel[1], rel[0], 0]) / orbit_radius
            m.put_start_and_end_on(pos, pos + tangent * 0.9)
            v_tag.move_to(m.get_end() + tangent * 0.25)

        def update_f(m):
            pos = vector_bot.get_center()
            radial_in = (center_orbit - pos) / orbit_radius
            m.put_start_and_end_on(pos, pos + radial_in * 0.9)
            f_tag.move_to(m.get_end() + radial_in * 0.25)

        vector_bot.add_updater(update_bot)
        v_arrow.add_updater(update_v)
        f_arrow.add_updater(update_f)
        
        self.play(FadeIn(b_field), FadeIn(b_label))
        self.add(vector_bot, v_arrow, f_arrow, v_tag, f_tag)
        
        self.wait(6) # Observe the orbit
        
        vector_bot.remove_updater(update_bot)
        v_arrow.remove_updater(update_v)
        f_arrow.remove_updater(update_f)
        self.wait(1)