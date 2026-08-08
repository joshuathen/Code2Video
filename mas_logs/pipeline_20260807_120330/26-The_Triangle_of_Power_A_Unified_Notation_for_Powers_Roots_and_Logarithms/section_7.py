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

class Section7Scene(TeachingScene):
    def construct(self):
        # setup_layout for Section 7
        self.setup_layout("The Unified Summary", [
            "All three operations live within this single triangle.",
            "Rotating our focus reveals powers, roots, or logs.",
            "The numbers two, three, and eight never move.",
            "One consistent map for all exponential math.",
            "Mathematics becomes simpler when we see the connections."
        ])

        # Assets
        triangle_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg"

        # Colors for vertices
        color_base = BLUE_B
        color_exp = GREEN_B
        color_res = RED_B
        
        # === Animation for Lecture Line 1 ===
        # "All three operations live within this single triangle."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Load and place the Triangle of Power [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg]
        triangle = SVGMobject(triangle_path, color=WHITE).scale(1.5)
        self.place_in_area(triangle, "B3", "E5")
        
        # Values at vertices: 2 (Base), 3 (Exponent), 8 (Result)
        val_base = MathTex("2", color=color_base)
        val_exp = MathTex("3", color=color_exp)
        val_res = MathTex("8", color=color_res)
        
        # Position vertices based on a triangle centered in B3-E5
        self.place_at_grid(val_base, "E3", scale_factor=1.2)
        self.place_at_grid(val_exp, "B4", scale_factor=1.2)
        self.place_at_grid(val_res, "E5", scale_factor=1.2)
        
        self.play(Create(triangle))
        self.play(FadeIn(val_base), FadeIn(val_exp), FadeIn(val_res))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Rotating our focus reveals powers, roots, or logs."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Pulse the triangle edges [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg]
        self.play(
            triangle.animate.set_stroke(width=10, color=YELLOW),
            run_time=0.4,
            rate_func=there_and_back
        )
        
        # Operation labels
        op_power = MathTex("2^3 = 8", color=YELLOW_B).scale(0.8)
        op_root = MathTex(r"\sqrt[3]{8} = 2", color=ORANGE).scale(0.8)
        op_log = MathTex(r"\log_2(8) = 3", color=PURPLE_B).scale(0.8)
        
        # Position operation labels near triangle edges
        self.place_at_grid(op_power, "E4", scale_factor=0.8) # Bottom
        self.place_at_grid(op_root, "C3", scale_factor=0.8)  # Left
        self.place_at_grid(op_log, "C5", scale_factor=0.8)   # Right
        
        self.play(FadeIn(op_power), FadeIn(op_root), FadeIn(op_log))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "The numbers two, three, and eight never move."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Highlight symmetry by rotating slightly [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg]
        full_tri = VGroup(triangle, val_base, val_exp, val_res)
        self.play(
            Rotate(full_tri, angle=10*DEGREES, about_point=full_tri.get_center()),
            run_time=0.5,
            rate_func=wiggle
        )
        
        # Indicate values are constant
        self.play(
            Indicate(val_base, color=color_base),
            Indicate(val_exp, color=color_exp),
            Indicate(val_res, color=color_res)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "One consistent map for all exponential math."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Fade out operation labels before scaling
        self.play(FadeOut(op_power), FadeOut(op_root), FadeOut(op_log))
        
        # Scale triangle and vertices to fill the area [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg]
        self.play(
            full_tri.animate.scale(1.4).move_to(self.grid["C4"]),
            run_time=1.5
        )
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # "Mathematics becomes simpler when we see the connections."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Final emphasis pulse
        self.play(Indicate(full_tri, color=GOLD))
        self.wait(2)
        
        # Fade out all elements
        self.play(
            FadeOut(full_tri),
            FadeOut(self.lecture),
            FadeOut(self.title)
        )
        self.wait(2)
