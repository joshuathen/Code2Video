from manim import *

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
        title = "Summary and the 'Cramer Formula'"
        lines = [
            "Cramer's Rule links geometric areas to algebraic solutions.",
            "The solution exists if the base area is non-zero.",
            "This formula works for any number of dimensions.",
            "Algebra and geometry unite in this elegant solution.",
            "Understanding this bridge simplifies complex linear systems."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Present the general Cramer's Rule formula for xi in white #FFFFFF.
        self.lecture[0].set_color(YELLOW)
        # Split MathTex for easy access to the denominator
        formula_xi = MathTex(r"x_i", "=", r"\frac{\det(A_i)}{", r"\det(A)", r"}", color=WHITE)
        # Fixed: Issue 31 - scale and area adjustment
        self.place_in_area(formula_xi, 'A2', 'B5', scale_factor=1.0)
        self.play(Write(formula_xi))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Flash the denominator det(A) in red #FF0000 to show singularity.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        denom = formula_xi[3]
        self.play(Indicate(denom, color="#FF0000", scale_factor=1.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final drone [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg] animation showing it reaching the target W using x and y.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Setup vectors for visualization in the bottom-right grid area
        origin = self.grid["E2"]
        v1_raw = np.array([1.2, 0, 0])
        v2_raw = np.array([0.4, 1.0, 0])
        x_scalar, y_scalar = 1.3, 0.9
        w_raw = x_scalar * v1_raw + y_scalar * v2_raw
        
        vec_v1 = Arrow(origin, origin + v1_raw, buff=0, color="#00FF00")
        vec_v2 = Arrow(origin, origin + v2_raw, buff=0, color="#0000FF")
        vec_w = Arrow(origin, origin + w_raw, buff=0, color="#FF0000")
        
        label_v1 = MathTex(r"\vec{v}_1", color="#00FF00", font_size=20).next_to(origin + v1_raw, DOWN, buff=0.1)
        label_v2 = MathTex(r"\vec{v}_2", color="#0000FF", font_size=20).next_to(origin + v2_raw, LEFT, buff=0.1)
        label_w = MathTex(r"\vec{w}", color="#FF0000", font_size=20).next_to(origin + w_raw, UP, buff=0.1)
        
        # Fixed: Issue 20 - Integrate drone asset
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg", height=0.3, color=WHITE)
        drone.move_to(origin)
        drone_label = Text("Drone", font_size=14, color=WHITE).next_to(drone, UP, buff=0.05)
        drone_group = VGroup(drone, drone_label)
        
        self.play(
            FadeIn(VGroup(vec_v1, vec_v2, label_v1, label_v2)),
            GrowArrow(vec_w), 
            Write(label_w)
        )
        
        # Animate drone along components to show reachability
        self.play(FadeIn(drone_group))
        self.play(drone_group.animate.move_to(origin + x_scalar * v1_raw), run_time=1.0)
        self.play(drone_group.animate.move_to(origin + w_raw), run_time=1.0)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight the geometric-algebraic link with text 'Bridge'.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        bridge_text = Text("Bridge", color=WHITE, font_size=32)
        # Fixed: Issue 33 - position and scale adjustment
        self.place_at_grid(bridge_text, 'F4', scale_factor=0.8)
        
        # Adjust arrows for the new bridge position
        arrow1 = Arrow(formula_xi.get_bottom(), bridge_text.get_top(), color=WHITE, buff=0.1)
        arrow2 = Arrow(origin + w_raw, bridge_text.get_bottom(), color=WHITE, buff=0.1)
        
        self.play(Write(bridge_text), Create(arrow1), Create(arrow2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Fade out everything except the final formulas in white #FFFFFF.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        final_x = MathTex(r"x = \frac{\det(A_x)}{\det(A)}", color=WHITE)
        final_y = MathTex(r"y = \frac{\det(A_y)}{\det(A)}", color=WHITE)
        final_formulas = VGroup(final_x, final_y).arrange(DOWN, buff=0.8)
        # Fixed: Issue 32 - scale and area adjustment
        self.place_in_area(final_formulas, 'C2', 'E5', scale_factor=1.0)
        
        self.play(
            FadeOut(VGroup(formula_xi, vec_v1, vec_v2, vec_w, label_v1, label_v2, label_w, drone_group, bridge_text, arrow1, arrow2)),
            FadeIn(final_formulas)
        )
        self.wait(2)
