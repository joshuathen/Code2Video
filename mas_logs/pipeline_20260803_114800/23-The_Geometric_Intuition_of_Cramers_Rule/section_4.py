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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Replace vector A with target vector W.",
            "The new area is x times the original area.",
            "This geometric scaling reveals the value of x.",
            "Thus, x is the ratio of these two areas.",
            "This ratio is the essence of Cramer's Rule."
        ]
        self.setup_layout("The Ratio of Areas (Finding x)", lecture_lines)
        
        # Define colors
        color_a = BLUE
        color_b = RED
        color_w = GREEN
        color_para_ab = YELLOW
        color_para_wb = ORANGE
        color_text = WHITE

        # Coordinates
        vec_a_coords = np.array([1, 0.2, 0])
        vec_b_coords = np.array([0.4, 0.8, 0])
        x_val = 1.8
        y_val = 0.4
        vec_w_coords = x_val * vec_a_coords + y_val * vec_b_coords
        vec_xa_coords = x_val * vec_a_coords

        # Helper to create parallelogram
        def get_parallelogram(v1, v2, color, fill_opacity=0.3):
            return Polygon(
                ORIGIN, v1, v1 + v2, v2,
                stroke_width=2, stroke_color=color, fill_color=color, fill_opacity=fill_opacity
            )

        # === Animation for Lecture Line 1 ===
        # Replace vector A with target vector W.
        self.lecture[0].set_color(color_w)
        
        # Top half: Parallelogram(A, B)
        v_a = Arrow(ORIGIN, vec_a_coords, buff=0, color=color_a)
        v_b = Arrow(ORIGIN, vec_b_coords, buff=0, color=color_b)
        para_ab = get_parallelogram(vec_a_coords, vec_b_coords, color_para_ab)
        group_ab = VGroup(para_ab, v_a, v_b)
        # Fix Issue 25: Reposition group_ab
        self.place_in_area(group_ab, "A2", "C4", scale_factor=0.9)
        
        label_ab = MathTex(r"\text{Area}(A, B)", font_size=28, color=color_para_ab)
        # Fix Issue 26: Rescale label_ab
        self.place_at_grid(label_ab, "A5", scale_factor=0.8)

        # Bottom half: Start with A, B
        v_a_bot = Arrow(ORIGIN, vec_a_coords, buff=0, color=color_a)
        v_b_bot = Arrow(ORIGIN, vec_b_coords, buff=0, color=color_b)
        para_ab_bot = get_parallelogram(vec_a_coords, vec_b_coords, color_para_ab)
        group_ab_bot = VGroup(para_ab_bot, v_a_bot, v_b_bot)
        # Fix Issue 25: Reposition group_ab_bot
        self.place_in_area(group_ab_bot, "D2", "F4", scale_factor=0.9)

        # Prepare W components
        v_w = Arrow(ORIGIN, vec_w_coords, buff=0, color=color_w)
        v_b_copy = Arrow(ORIGIN, vec_b_coords, buff=0, color=color_b)
        para_wb = get_parallelogram(vec_w_coords, vec_b_coords, color_para_wb)
        group_wb = VGroup(para_wb, v_w, v_b_copy)
        # Fix Issue 25: Reposition group_wb
        self.place_in_area(group_wb, "D2", "F4", scale_factor=0.9)

        label_wb = MathTex(r"\text{Area}(W, B)", font_size=28, color=color_para_wb)
        # Fix Issue 26: Rescale label_wb
        self.place_at_grid(label_wb, "D5", scale_factor=0.8)

        self.play(FadeIn(group_ab), Write(label_ab))
        self.play(FadeIn(group_ab_bot))
        self.wait(1)
        
        # Replace A with W in the bottom view
        self.play(Transform(group_ab_bot, group_wb), Write(label_wb))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The new area is x times the original area.
        self.lecture[1].set_color(color_para_wb)
        
        # Transform Parallelogram(W, B) to Parallelogram(xA, B)
        para_xab = get_parallelogram(vec_xa_coords, vec_b_coords, color_para_wb)
        v_xa = Arrow(ORIGIN, vec_xa_coords, buff=0, color=color_a)
        
        group_xab_target = VGroup(para_xab, v_xa, Arrow(ORIGIN, vec_b_coords, buff=0, color=color_b))
        self.place_in_area(group_xab_target, "D2", "F4", scale_factor=0.9)

        self.play(
            Transform(group_ab_bot[0], group_xab_target[0]),
            Transform(group_ab_bot[1], group_xab_target[1]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This geometric scaling reveals the value of x.
        self.lecture[2].set_color(color_a)
        
        # Overlay original Area(A, B) to visualize the x-scaling
        para_ab_ghost = para_ab.copy().set_color(color_para_ab).set_stroke(opacity=0.5).set_fill(opacity=0.1)
        shift_vec = group_ab_bot[1].get_start() - para_ab_ghost.get_vertices()[0]
        para_ab_ghost.shift(shift_vec)

        self.play(FadeIn(para_ab_ghost))
        self.play(Indicate(group_ab_bot[1], color=color_a))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Thus, x is the ratio of these two areas.
        self.lecture[3].set_color(color_text)
        
        ratio_eqn = MathTex(r"x = \frac{\text{Area}(W, B)}{\text{Area}(A, B)}", font_size=28, color=color_text)
        # Fix Issue 27: Rescale ratio_eqn
        self.place_at_grid(ratio_eqn, "E5", scale_factor=0.8)
        
        self.play(Write(ratio_eqn))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This ratio is the essence of Cramer's Rule.
        self.lecture[4].set_color(color_text)
        
        det_eqn = MathTex(r"x = \frac{\det(W, B)}{\det(A, B)}", font_size=28, color=color_text)
        # Fix Issue 27: Rescale det_eqn
        self.place_at_grid(det_eqn, "F5", scale_factor=0.8)

        self.play(FadeIn(det_eqn, shift=DOWN))
        self.wait(2)
