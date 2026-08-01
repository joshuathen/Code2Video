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
        title_text = "The Gateway to Linear Algebra: Linear Combinations"
        lines = [
            "Linear combinations involve scaling and adding multiple vectors.",
            "We scale each vector first, then add them together.",
            "This process allows us to reach any point in space.",
            "It is the core concept behind basis vectors and matrices.",
            "These combinations form the foundation of all linear algebra."
        ]
        
        self.setup_layout(title_text, lines)
        
        # Colors
        u_color = "#00BFFF"
        v_color = "#FF69B4"
        res_color = "#F0E68C"
        
        # Grid positions for vectors
        # Issue 26: Vector origin at E2
        origin_pos = self.grid["E2"]
        u_tip_initial = self.grid["E4"]
        v_tip_initial = self.grid["D2"]
        
        # Asset preparation
        foundation_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/foundation.svg")
        gateway_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gateway.svg")
        
        # Scale factors
        a_tracker = ValueTracker(1.0)
        b_tracker = ValueTracker(1.0)

        # === Animation for Lecture Line 1 ===
        # Display base vectors U and V, incorporating foundation asset.
        self.play(self.lecture[0].animate.set_color(u_color))
        
        u_vec = Arrow(origin_pos, u_tip_initial, color=u_color, buff=0)
        v_vec = Arrow(origin_pos, v_tip_initial, color=v_color, buff=0)
        u_label = MathTex(r"\vec{u}", color=u_color, font_size=24).next_to(u_tip_initial, DOWN, buff=0.1)
        v_label = MathTex(r"\vec{v}", color=v_color, font_size=24).next_to(v_tip_initial, LEFT, buff=0.1)
        
        self.place_at_grid(foundation_icon, "F2", scale_factor=0.5)
        
        self.play(Create(u_vec), Write(u_label), FadeIn(foundation_icon))
        self.play(Create(v_vec), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Scale Vector U by factor 'a' visually.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(u_color)
        )
        
        u_label_scaled = MathTex(r"a\vec{u}", color=u_color, font_size=24)
        
        def update_u(m):
            # Scale factor moves tip from E4 towards E6
            new_tip = origin_pos + (u_tip_initial - origin_pos) * a_tracker.get_value()
            m.put_start_and_end_on(origin_pos, new_tip)
            
        def update_u_label(m):
            new_tip = origin_pos + (u_tip_initial - origin_pos) * a_tracker.get_value()
            m.next_to(new_tip, DOWN, buff=0.1)

        u_vec.add_updater(update_u)
        u_label.add_updater(update_u_label)
        
        self.play(
            a_tracker.animate.set_value(2.0), # Reaches E6
            Transform(u_label, u_label_scaled),
            run_time=2
        )
        u_vec.remove_updater(update_u)
        u_label.remove_updater(update_u_label)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Scale Vector V by factor 'b' visually.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(v_color)
        )
        
        # Issue 27: v_label at B2
        v_label_scaled = MathTex(r"b\vec{v}", color=v_color, font_size=24)
        self.place_at_grid(v_label_scaled, "B2", scale_factor=0.8)

        def update_v(m):
            # Scale factor moves tip from D2 towards B2
            new_tip = origin_pos + (v_tip_initial - origin_pos) * b_tracker.get_value()
            m.put_start_and_end_on(origin_pos, new_tip)

        v_vec.add_updater(update_v)
        
        self.play(
            b_tracker.animate.set_value(3.0), # Reaches B2
            FadeTransform(v_label, v_label_scaled),
            run_time=2
        )
        v_vec.remove_updater(update_v)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Place scaled V at the tip of scaled U.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        u_end = u_vec.get_end()
        v_vec_offset = v_vec.get_end() - v_vec.get_start()
        v_final_tip = u_end + v_vec_offset
        
        self.play(
            v_vec.animate.move_to(u_end + v_vec_offset/2),
            v_label_scaled.animate.next_to(v_final_tip, UP, buff=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Draw resultant vector 'aU + bV' transitioning through gateway asset.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(res_color)
        )
        
        resultant_arrow = Arrow(origin_pos, v_final_tip, color=res_color, buff=0)
        resultant_label = MathTex(r"a\vec{u} + b\vec{v}", color=res_color, font_size=28)
        
        # Issue 25: resultant_label in B4-C6
        self.place_in_area(resultant_label, "B4", "C6", scale_factor=0.7)
        
        # Gateway icon at A6
        self.place_at_grid(gateway_icon, "A6", scale_factor=0.5)
        
        self.play(
            Create(resultant_arrow), 
            Write(resultant_label), 
            FadeIn(gateway_icon)
        )
        self.wait(3)
