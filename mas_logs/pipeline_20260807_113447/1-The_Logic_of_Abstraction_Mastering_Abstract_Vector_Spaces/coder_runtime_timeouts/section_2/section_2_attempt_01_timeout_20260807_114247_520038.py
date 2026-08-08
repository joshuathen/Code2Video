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
        # Setup the scene with title and lecture lines
        self.setup_layout(
            "Prerequisite Check: The Two Pillars",
            [
                "Two fundamental operations support everything.",
                "Addition combines vectors tip-to-tail.",
                "Scalar multiplication scales their magnitude."
            ]
        )
        
        # Set initial colors for lecture lines (dimmed)
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create a background grid for the operations
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "stroke_width": 2},
            background_line_style={"stroke_opacity": 0.2, "stroke_width": 1}
        )
        # Place it in the central visual area on the right
        self.place_in_area(plane, 'A1', 'F6')
        
        self.play(Create(plane), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight current line, dim previous
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Define colors for vectors
        u_color = "#E74C3C" # Red
        v_color = "#3498DB" # Blue
        sum_color = "#F1C40F" # Yellow
        
        # Vector Addition: Tip-to-Tail
        u_vec = Arrow(plane.c2p(0, 0), plane.c2p(1, 1), buff=0, color=u_color)
        v_vec = Arrow(plane.c2p(1, 1), plane.c2p(2.5, 0.5), buff=0, color=v_color)
        sum_vec = Arrow(plane.c2p(0, 0), plane.c2p(2.5, 0.5), buff=0, color=sum_color)
        
        u_label = MathTex("\\vec{u}", color=u_color).scale(0.8)
        u_label.next_to(u_vec.get_center(), UL, buff=0.1)
        
        v_label = MathTex("\\vec{v}", color=v_color).scale(0.8)
        v_label.next_to(v_vec.get_center(), UR, buff=0.1)
        
        sum_label = MathTex("\\vec{u} + \\vec{v}", color=sum_color).scale(0.8)
        sum_label.next_to(sum_vec.get_center(), DOWN, buff=0.2)

        self.play(GrowArrow(u_vec), Write(u_label))
        self.wait(0.5)
        self.play(GrowArrow(v_vec), Write(v_label))
        self.wait(1)
        self.play(GrowArrow(sum_vec), Write(sum_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight current line, dim previous
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Clear addition vectors to focus on scalar multiplication
        self.play(FadeOut(u_vec, v_vec, sum_vec, u_label, v_label, sum_label))
        
        # Scalar Multiplication
        w_color = "#9B59B6" # Purple
        res_color = "#F39C12" # Orange/Yellow
        
        # Base vector
        w_start = plane.c2p(-1, 0)
        w_end_1 = plane.c2p(0.5, 0)
        w_end_2 = plane.c2p(2, 0)
        
        w_vec = Arrow(w_start, w_end_1, buff=0, color=w_color)
        w_label = MathTex("\\vec{w}", color=w_color).scale(0.8)
        w_label.next_to(w_vec, UP, buff=0.1)
        
        self.play(GrowArrow(w_vec), Write(w_label))
        self.wait(1)
        
        # Scalar tracker for stretching animation
        scalar = ValueTracker(1.0)
        
        # Persistent mobject to stretch
        # We use add_updater to change properties in place
        def update_w(m):
            s = scalar.get_value()
            new_end = plane.c2p(-1 + 1.5 * s, 0)
            m.put_start_and_end_on(w_start, new_end)
            m.set_color(interpolate_color(w_color, res_color, s - 1))

        w_vec.add_updater(update_w)
        
        # Prepare label transform
        target_label = MathTex("2\\vec{w}", color=res_color).scale(0.8)
        target_label.next_to(w_end_2, UP, buff=0.1)
        
        self.play(
            scalar.animate.set_value(2.0),
            w_label.animate.become(target_label),
            run_time=2,
            rate_func=smooth
        )
        self.wait(2)

        # Final cleanup for the section
        w_vec.remove_updater(update_w)
        self.wait(1)
