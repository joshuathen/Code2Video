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
        # Initial Setup
        title = "The Advanced View: Cross Product as a Linear Transformation"
        lines = [
            'View the cross product as a linear transformation.',
            'It maps a vector w to a scalar volume.',
            'This transformation is represented by a unique dual vector.',
            'The cross product u cross v is that vector.',
            'It maximizes the volume when w is perpendicular.'
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_U = BLUE_B
        COLOR_V = RED_B
        COLOR_W = GREEN_B
        COLOR_DUAL = YELLOW_B
        COLOR_VOL = PURPLE_A

        # Coordinate definitions for mock-3D (on 2D plane)
        origin = self.grid["D4"]
        u_dir = np.array([1.2, -0.2, 0])
        v_dir = np.array([0.6, 0.4, 0])
        w_dir_perp = np.array([0, 1.2, 0]) # Perpendicular to u and v in projection
        w_angle = ValueTracker(-0.5) # radians offset from perpendicular

        def get_w_vec():
            ang = w_angle.get_value()
            # Rotate w_dir_perp around origin by ang in the XY plane
            rot_mat = np.array([
                [np.cos(ang), -np.sin(ang), 0],
                [np.sin(ang), np.cos(ang), 0],
                [0, 0, 1]
            ])
            return np.dot(rot_mat, w_dir_perp)

        def create_parallelepiped(o, u, v, w, color, opacity=0.3):
            pts = [
                o, o + u, o + u + v, o + v,
                o + w, o + u + w, o + u + v + w, o + v + w
            ]
            faces_indices = [
                [0, 1, 2, 3], [4, 5, 6, 7], # Bottom, Top
                [0, 1, 5, 4], [1, 2, 6, 5], # Sides
                [2, 3, 7, 6], [3, 0, 4, 7]
            ]
            faces = VGroup()
            for idxs in faces_indices:
                faces.add(Polygon(*[pts[i] for i in idxs], fill_opacity=opacity, fill_color=color, stroke_width=0.5, stroke_color=WHITE))
            return faces

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_U))
        vec_u = Arrow(origin, origin + u_dir, buff=0, color=COLOR_U)
        vec_v = Arrow(origin, origin + v_dir, buff=0, color=COLOR_V)
        label_u = Text("u", color=COLOR_U, font_size=20, slant=ITALIC)
        label_v = Text("v", color=COLOR_V, font_size=20, slant=ITALIC)
        self.place_at_grid(label_u, "E5")
        self.place_at_grid(label_v, "C5")
        
        self.play(Create(vec_u), Create(vec_v), Write(label_u), Write(label_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_W))
        
        current_w = get_w_vec()
        vec_w = Arrow(origin, origin + current_w, buff=0, color=COLOR_W)
        label_w = Text("w", color=COLOR_W, font_size=20, slant=ITALIC)
        label_w.move_to(origin + current_w * 1.2)
        
        para = create_parallelepiped(origin, u_dir, v_dir, current_w, COLOR_W)
        
        self.play(Create(vec_w), Write(label_w))
        self.play(FadeIn(para))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_VOL))
        formula = Text("T(w) = Vol(u, v, w)", color=COLOR_VOL, font_size=24)
        self.place_in_area(formula, "B2", "C3", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_DUAL))
        vec_dual = Arrow(origin, origin + w_dir_perp * 1.2, buff=0, color=COLOR_DUAL)
        label_dual = Text("u × v", color=COLOR_DUAL, font_size=20, slant=ITALIC)
        self.place_at_grid(label_dual, "B5")
        
        self.play(Create(vec_dual), Write(label_dual))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_W))
        
        vol_text = Text("Vol = ", color=WHITE, font_size=20)
        vol_val = DecimalNumber(0, color=COLOR_W, font_size=20, num_decimal_places=2)
        vol_group = VGroup(vol_text, vol_val).arrange(RIGHT, buff=0.1)
        self.place_at_grid(vol_group, "E6")
        self.add(vol_group)

        # Updaters for interactivity
        def update_para(mob):
            new_w = get_w_vec()
            mob.become(create_parallelepiped(origin, u_dir, v_dir, new_w, COLOR_W))

        def update_vec_w(mob):
            new_w = get_w_vec()
            mob.put_start_and_end_on(origin, origin + new_w)

        def update_label_w(mob):
            new_w = get_w_vec()
            mob.move_to(origin + new_w * 1.2)

        def update_vol_val(mob):
            # Volume is dot product of w with the normalized dual vector in this 2D representation
            val = abs(np.dot(get_w_vec(), w_dir_perp / np.linalg.norm(w_dir_perp)))
            mob.set_value(val)

        para.add_updater(update_para)
        vec_w.add_updater(update_vec_w)
        label_w.add_updater(update_label_w)
        vol_val.add_updater(update_vol_val)

        # Swings w to demonstrate volume maximization
        self.play(w_angle.animate.set_value(0.6), run_time=1.5, rate_func=smooth)
        self.play(w_angle.animate.set_value(0.0), run_time=1.5, rate_func=smooth)
        self.play(w_angle.animate.set_value(-0.6), run_time=1.5, rate_func=smooth)
        self.play(w_angle.animate.set_value(0.0), run_time=1.5, rate_func=smooth)
        
        self.wait(2)